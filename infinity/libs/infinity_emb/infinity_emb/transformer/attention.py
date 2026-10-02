# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""Attention implementation selection for torch-engine models.

Torch rerankers can be loaded with ``attn_implementation="eager" | "sdpa" |
"flash_attention_2"``. Flash attention 2 computes attention without
materializing dense ``[B, 1, L, L]`` masks (padding is handled in-kernel), so
memory drops to ``O(B * L)`` for long listwise rerank prompts. Flash attention
is not a packaging extra (operator-side install of ``flash-attn``) and its
kernels only support fp16/bf16, so requested-but-unsatisfiable choices degrade
*loudly* to sdpa via :func:`resolve_attn_implementation`.
"""

from __future__ import annotations

from typing import Optional

from infinity_emb._optional_imports import CHECK_FLASH_ATTN, CHECK_TORCH
from infinity_emb.log_handler import logger

if CHECK_TORCH.is_available:
    import torch
else:

    class torch:  # type: ignore[no-redef]
        pass

#: values accepted by the ``attn_implementation`` knob (``None`` = model default)
ATTN_IMPLEMENTATIONS = ("eager", "sdpa", "flash_attention_2")


def resolve_attn_implementation(requested: Optional[str], loading_dtype) -> Optional[str]:
    """Return the attn_implementation to pass to the model, or None (model default).

    Args:
        requested: user-selected value from ``EngineArgs.attn_implementation``;
            ``None`` means model default.
        loading_dtype: the torch dtype the model will be loaded in
            (``ls.loading_dtype``); ``None`` (auto) is treated as acceptable,
            since auto resolves to bf16/fp16 on GPU.

    Returns:
        The value to pass to the model. ``None``/``eager``/``sdpa`` pass
        through untouched. ``"flash_attention_2"`` is only honored when the
        ``flash_attn`` package is installed AND the loading dtype is not
        float32 (FA2 kernels are fp16/bf16 only); otherwise it degrades to
        ``"sdpa"`` with a visible warning.
    """
    if requested not in ATTN_IMPLEMENTATIONS:
        return None
    if requested != "flash_attention_2":
        return requested
    if not CHECK_FLASH_ATTN.is_available:
        logger.warning(
            "attn_implementation=\"flash_attention_2\" requested but the "
            "`flash-attn` package is not installed (pip install flash-attn); "
            "falling back to sdpa."
        )
        return "sdpa"
    if loading_dtype is not None and loading_dtype == torch.float32:
        logger.warning(
            "attn_implementation=\"flash_attention_2\" requires fp16/bf16 "
            "kernels, but the model is loading in float32 (jina cosine scores "
            "drift ~1e-3 in bf16, ordering unaffected); falling back to sdpa."
        )
        return "sdpa"
    return "flash_attention_2"


def verify_attn_implementation(model, requested: Optional[str]) -> None:
    """Warn if the loaded model's effective attention implementation != requested.

    Remote-code models (e.g. the vendored ``JinaForRanking``) may pin their
    own implementation and silently ignore the requested one. Reads
    ``model.config._attn_implementation`` (falling back to
    ``model.model.config``/``model.transformer.config`` for wrapped
    containers) and warns on any mismatch.
    """
    if requested is None:
        return
    config = getattr(model, "config", None)
    if config is None:
        # wrapped containers (e.g. sentence-transformers CrossEncoder)
        for attr in ("model", "transformer"):
            config = getattr(getattr(model, attr, None), "config", None)
            if config is not None:
                break
    effective = None if config is None else getattr(config, "_attn_implementation", None)
    if effective is None:
        logger.debug(
            "could not determine the loaded model's effective attention "
            f"implementation (requested {requested!r}); skipping verification."
        )
        return
    if effective != requested:
        logger.warning(
            f"attention implementation mismatch: requested {requested!r} but the "
            f"model loaded with {effective!r}. Some remote/quantized "
            "implementations pin their own attention; scores are still valid, "
            "but the memory profile of flash attention is not active."
        )
    else:
        logger.info(f"attention implementation {effective!r} active after load.")
