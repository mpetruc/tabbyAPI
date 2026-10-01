import numpy as np
import pytest
import torch

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)

try:
    from fastembed import TextEmbedding  # type: ignore

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


@pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
def test_sentence_transformer_equals_fastembed(
    text=(
        "This is a test sentence for the sentence transformer and fastembed. The PyTorch API of nested tensors is in prototype stage and will change in the near future."
    ),
    model_name="BAAI/bge-small-en-v1.5",
) -> None:
    model_st = SentenceTransformerPatched(engine_args=EngineArgs(model_name_or_path=model_name))
    model_fastembed = TextEmbedding(model_name)

    embedding_st = model_st.encode_post(model_st.encode_core(model_st.encode_pre([text])))
    embedding_fast = np.array(list(model_fastembed.embed(documents=[text])))

    assert embedding_fast.shape == embedding_st.shape
    assert np.allclose(embedding_st[0], embedding_fast[0], atol=1e-3)
    # cosine similarity
    sim = torch.nn.functional.cosine_similarity(
        torch.tensor(embedding_st), torch.tensor(embedding_fast)
    )
    assert sim > 0.99


import os

NEEDS_DOWNLOAD = os.getenv("JINA_DOWNLOAD_TESTS", "0") == "1"


@pytest.mark.skipif(
    not NEEDS_DOWNLOAD,
    reason="set JINA_DOWNLOAD_TESTS=1 to run model-backed tests (downloads ~1.2 GB)",
)
def test_sentence_transformer_qwen3_embedding():
    """Qwen3-Embedding models are decoder-based sentence-transformers; they
    must load and embed through the ST path on the pinned matrix
    (transformers 4.57.x + sentence-transformers 3.3.x)."""
    model = SentenceTransformerPatched(
        engine_args=EngineArgs(
            model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
            model_warmup=False,
        )
    )
    embeddings = model.encode(["Hello world", "Qwen3 supports embeddings"])
    assert embeddings.shape == (2, 1024)
