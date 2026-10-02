# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""Torch engine for the LBNL listwise reranker family (``JinaForRanking``).

Serves ``jinaai/jina-reranker-v3`` and ``jinaai/jina-reranker-v3.5`` through
the standard rerank pipeline. Two scoring paths are provided:

- **Option A (pairwise, default):** one verbatim prompt per ``(query, doc)``
  pair. Scores are the model's own cosine similarities and are exposed as
  ``score_range = "cosine"`` so ``BatchHandler`` does not sigmoid them.
  Single-doc relevance only; ordering is meaningful, cross-document
  normalization is not applied (see feasibility report §6, Option A vs B).
- **Option B (listwise, ``rerank_listwise=True``):** one prompt per block of
  documents with the remote implementation's block flushing, per-block
  max-normalized weights and request-global weighted query embedding — the
  faithful scoring path. Used by ``BatchHandler.rerank`` for the whole
  request.

Design constraints (see docs/jina-reranker-family-feasibility.md §3):
- The prompt layout is load-bearing; it is reproduced via the remote module's
  ``format_docs_prompts_func`` verbatim — never ``apply_chat_template``.
- The remote ``forward()`` locates the readout tokens (``<|embed_token|>``
  151670 / ``<|rerank_token|>`` 151671) internally and returns ``scores`` with
  ``logits=None``; ``use_cache=False`` is forced by the remote code.
- Loading must go through ``AutoModel.from_pretrained(trust_remote_code=True)``;
  ``AutoModelForSequenceClassification`` cannot resolve the remote
  ``auto_map = {"AutoModel": "modeling.JinaForRanking"}`` registration.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

from infinity_emb._optional_imports import (
    CHECK_SENTENCE_TRANSFORMERS,
    CHECK_TORCH,
    CHECK_TRANSFORMERS,
)
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.transformer._compat import from_pretrained_dtype_kwarg
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.attention import (
    resolve_attn_implementation,
    verify_attn_implementation,
)

if CHECK_TORCH.is_available and CHECK_TRANSFORMERS.is_available:
    import torch
    from transformers import AutoModel, AutoTokenizer
else:

    class torch:  # type: ignore[no-redef]
        pass


if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "JinaV3CrossEncoder",
]

#: arch name matched exactly in `inference/select_model.py`
JINA_V3_ARCH = "JinaForRanking"


class JinaV3CrossEncoder(BaseCrossEncoder):
    """CrossEncoder-style adapter for `JinaForRanking` (qwen3, LBNL) models."""

    capabilities = {"rerank"}
    #: scores are cosine similarities in [-1, 1]; BatchHandler must not sigmoid
    score_range = "cosine"

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()

        if engine_args.bettertransformer:
            logger.warning(
                "JinaForRanking models run with the default (sdpa) attention "
                "implementation; `bettertransformer` is ignored for this family."
            )

        ls = engine_args._loading_strategy
        assert ls is not None

        model_kwargs = {}
        if ls.loading_dtype is not None:  # type: ignore[attr-defined]
            model_kwargs[from_pretrained_dtype_kwarg()] = ls.loading_dtype
        resolved_attn = resolve_attn_implementation(
            engine_args.attn_implementation, ls.loading_dtype
        )
        if resolved_attn is not None:
            model_kwargs["attn_implementation"] = resolved_attn
        logger.info(
            "attention implementation for %s: %s",
            engine_args.model_name_or_path,
            resolved_attn or "model default",
        )

        self.model = AutoModel.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            **model_kwargs,
        )
        self.model.to(ls.device_placement)
        self.model.eval()
        verify_attn_implementation(self.model, resolved_attn)

        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            # Jina v3/v3.5 ship a Mistral-Small-3.1-24B-derived tokenizer with
            # the known-buggy pre-tokenizer regex baked into tokenizer.json.
            # transformers >= 4.57 warns about it and replaces the pattern only
            # when this flag is set (on older versions it is inert).
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

        # copy of the tokenizer for cross-thread length counting (safe)
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

        # Option B knobs
        self.rerank_passages_per_block = int(
            getattr(engine_args, "rerank_passages_per_block", 16)
        )
        self._rerank_listwise = bool(getattr(engine_args, "rerank_listwise", False))

        # verbatim prompt builder, resolved from the loaded remote module
        self._prompt_fn = self._resolve_remote_fn("format_docs_prompts_func")

        # per-family truncation caps: reranker-v3 uses (query 512, doc 2048)
        # as signature defaults; reranker-v3.5 hard-codes (1024, 8192) in the
        # remote body. Resolve from the loaded code so the listwise path is
        # faithful on both.
        self._max_query_length, self._max_doc_length = self._resolve_truncation_caps()

        if ls.loading_dtype in (torch.float16, torch.bfloat16):
            logger.warning(
                "JinaForRanking scores are cosine similarities; loading in "
                f"{ls.loading_dtype} drifts them by ~1e-3 vs float32 "
                "(order is unaffected). Use dtype=float32 for golden-grade "
                "score reproducibility across devices."
            )

        # request-grouped listwise calls share the (CPU) model; serialize them
        self._listwise_lock = threading.Lock()

    def _resolve_remote_fn(self, name: str):
        """Fetch a module-level helper from the loaded remote code module."""
        module = importlib.import_module(type(self.model).__module__)
        return getattr(module, name)

    def _resolve_truncation_caps(self) -> tuple[int, int]:
        """Determine the reference (max_query_length, max_doc_length).

        Reads signature defaults first (reranker-v3 exposes them as
        parameters); otherwise extracts the literals from the remote body
        (reranker-v3.5 hard-codes 1024/8192). Falls back to (512, 2048).
        """
        caps: set[int] = set()
        rerank = getattr(self.model.rerank, "__wrapped__", self.model.rerank)
        try:
            sig = inspect.signature(rerank)
            for key in ("max_query_length", "max_doc_length"):
                default = sig.parameters[key].default if key in sig.parameters else None
                if isinstance(default, int) and default > 0:
                    caps.add(default)
        except (ValueError, TypeError):
            pass
        for const in rerank.__code__.co_consts:
            if isinstance(const, int) and 128 <= const <= 1 << 20:
                caps.add(const)
        query_vals = [c for c in caps if 128 <= c <= 2048]
        doc_vals = [c for c in caps if c >= 2048]
        query = query_vals[0] if query_vals else 512
        doc = max(doc_vals) if doc_vals else 2048
        logger.debug(
            "resolved JinaForRanking truncation caps: "
            f"max_query_length={query}, max_doc_length={doc}"
        )
        return query, doc

    def _build_prompt(self, query: str, document: str) -> str:
        """Reproduce the remote prompt layout verbatim for a single document."""
        return self._prompt_fn(
            query,
            [document],
            instruction=None,
            special_tokens=self.model.special_tokens,  # type: ignore[attr-defined]
            no_thinking=True,
        )

    def encode_pre(self, input_tuples: list[tuple[str, str]]):
        # return tokenized prompts (one prompt per pair)
        prompts = [
            self._build_prompt(query.strip(), document.strip()) for query, document in input_tuples
        ]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return tokenized

    def encode_core(self, features: dict[str, "Tensor"]):
        """Single forward per batch; readout is handled by the remote code.

        Returns the model's own cosine scores, shape (batch, 1).
        """
        with torch.no_grad():
            features = {k: v.to(self.model.device) for k, v in features.items()}
            out_features = self.model(**features, return_dict=True)
        return out_features.scores.detach().cpu()

    def encode_post(self, scores) -> list[float]:
        return scores.flatten().to(torch.float32).numpy()

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]

    # ------------------------------------------------------------------ #
    # Option B: faithful listwise scoring (mirrors the remote `rerank()`) #
    # ------------------------------------------------------------------ #

    def rerank_list(
        self,
        query: str,
        documents: list[str],
        passages_per_block: int = 16,
        top_n: int = 128,
        max_doc_length: Optional[int] = None,
        max_query_length: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
        """Score a full candidate list with the model's block logic.

        Mirrors the remote ``JinaForRanking.rerank`` implementation
        (block flushing by ``model_max_length - 2*query_length``, per-block
        max-normalized weights, request-global weighted-average of the query
        embedding, cosine scores) so the output is faithful to the model's
        design. ``passages_per_block`` default 16 matches the calibration
        range of the paper (the remote hard-codes 125, beyond it).

        Returns:
            (indices_sorted, scores_sorted): descending cosine similarity,
            ``scores_sorted`` in [-1, 1] aligned to the *input* document list
            via ``indices_sorted``.
        """
        if not self._rerank_listwise:
            raise RuntimeError(
                "rerank_list requires EngineArgs(rerank_listwise=True) for this model."
            )
        max_doc_length = max_doc_length or self._max_doc_length
        max_query_length = max_query_length or self._max_query_length
        with self._listwise_lock:
            order, scores = self._rerank_list_unlocked(
                query,
                documents,
                max(1, int(passages_per_block)),
                top_n,
                max_doc_length,
                max_query_length,
            )
        return order, scores

    @torch.no_grad()
    def _rerank_list_unlocked(
        self,
        query: str,
        documents: list[str],
        passages_per_block: int,
        top_n: int,
        max_doc_length: int,
        max_query_length: int,
    ) -> tuple[list[int], list[float]]:
        tokenizer = self._infinity_tokenizer
        max_length = tokenizer.model_max_length

        query, docs, doc_lengths, query_length = self.model._truncate_texts(  # type: ignore[attr-defined]
            query, documents, max_query_length, max_doc_length
        )

        length_capacity = max_length - 2 * query_length
        block_docs: list[str] = []
        doc_embeddings: list[list[float]] = []
        query_embeddings: list[list[float]] = []
        block_weights: list[float] = []

        def flush_block() -> None:
            nonlocal length_capacity
            outputs = self.model._compute_single_batch(query, block_docs, instruction=None)  # type: ignore[attr-defined]
            doc_embeddings.extend(outputs.doc_embeds[0].cpu().float().numpy())
            query_embeddings.append(outputs.query_embeds[0].cpu().float().numpy())
            scores = outputs.scores.view(-1).cpu().float().numpy()
            block_weights.append(((1.0 + scores) / 2.0).max())
            block_docs.clear()
            length_capacity = max_length - 2 * query_length

        for length, doc in zip(doc_lengths, docs):
            block_docs.append(doc)
            length_capacity -= length
            if len(block_docs) >= passages_per_block or length_capacity <= max_doc_length:
                flush_block()

        if block_docs:
            flush_block()

        query_embeddings_arr = np.array(query_embeddings)
        doc_embeddings_arr = np.array(doc_embeddings)
        query_embeddings_arr = np.average(
            query_embeddings_arr, axis=0, weights=np.array(block_weights)
        )

        scores = self.model._calculate_cosine_scores(  # type: ignore[attr-defined]
            query_embeddings_arr, doc_embeddings_arr
        )[0]

        order = np.argsort(scores)[::-1]
        if top_n is not None and top_n > 0:
            order = order[: min(top_n, len(order))]
        return order.tolist(), scores[order].tolist()
