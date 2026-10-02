import gc
import pathlib
import torch
from loguru import logger
from typing import List, Optional

from common.utils import unwrap
from common.optional_dependencies import dependencies

# Conditionally import infinity to sidestep its logger
if dependencies.extras:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine


class InfinityContainer:
    model_dir: pathlib.Path
    loaded: bool = False

    # Use a runtime type hint here
    engine: Optional["AsyncEmbeddingEngine"] = None

    def __init__(self, model_directory: pathlib.Path):
        self.model_dir = model_directory

    async def load(self, **kwargs):
        # Use cpu by default
        device = unwrap(kwargs.get("embeddings_device"), "cpu")

        engine_args = EngineArgs(
            model_name_or_path=str(self.model_dir),
            engine="torch",
            device=device,
            dtype=unwrap(kwargs.get("embeddings_dtype"), "auto"),
            bettertransformer=False,
            model_warmup=False,
            # JinaForRanking (reranker v3/v3.5) knobs: request-grouped
            # listwise scoring (Option B) and its block size. When False,
            # the default pairwise path is used (Option A).
            rerank_listwise=unwrap(kwargs.get("rerank_listwise"), False),
            rerank_passages_per_block=unwrap(
                kwargs.get("rerank_passages_per_block"), 16
            ),
            attn_implementation=unwrap(
                kwargs.get("embeddings_attn_implementation"), None
            ),
        )

        self.engine = AsyncEmbeddingEngine.from_args(engine_args)
        await self.engine.astart()

        self.loaded = True
        logger.info("Embedding model successfully loaded.")

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Embedding model unloaded.")

    async def generate(self, sentence_input: List[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}

    async def rerank(
        self,
        query: str,
        documents: List[str],
        raw_scores: bool = False,
        top_n: Optional[int] = None,
    ):
        """Rerank documents against a query.

        Score semantics follow the loaded model family: crossencoders report
        sigmoided logits for ``raw_scores=False``, while JinaForRanking
        engines (reranker v3/v3.5) report cosine similarities in [-1, 1]
        remapped to [0, 1]; ``raw_scores=True`` always returns the native
        scores (logits or cosine).
        """
        results, usage = await self.engine.rerank(
            query=query,
            docs=documents,
            raw_scores=raw_scores,
            top_n=top_n,
        )

        return {"results": results, "usage": usage}
