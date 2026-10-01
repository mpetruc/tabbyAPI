"""Model-backed tests for the JinaForRanking (LBNL) reranker engine.

Gated behind ``JINA_DOWNLOAD_TESTS=1`` because they download ~1.2 GB of
weights on first run. Run with:

    JINA_DOWNLOAD_TESTS=1 uv run python -m pytest \
        tests/unit_test/transformer/crossencoder/test_jina_v3.py -q
"""

import os

import numpy as np
import pytest

from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device
from infinity_emb.transformer.crossencoder.jina_v3 import JinaV3CrossEncoder

MODEL_ID = os.getenv("JINA_RERANKER_V3_MODEL", "jinaai/jina-reranker-v3")
REVISION = os.getenv("JINA_RERANKER_V3_REVISION", "d7d7e73b6ea138ced340b83865931b5dfb6c97aa")
DEVICE = getattr(Device, os.getenv("JINA_DEVICE", "cpu").upper(), Device.cpu)

needs_download = pytest.mark.skipif(
    os.getenv("JINA_DOWNLOAD_TESTS", "0") != "1",
    reason="set JINA_DOWNLOAD_TESTS=1 to run model-backed tests (downloads ~1.2 GB)",
)

QUERY = "What is the capital of France?"
DOCS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "France is a country in Western Europe; its capital is Paris.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    "Munich is the capital of Bavaria, Germany.",
]


@needs_download
@pytest.fixture(scope="module")
def engine():
    return JinaV3CrossEncoder(
        engine_args=EngineArgs(
            model_name_or_path=MODEL_ID,
            revision=REVISION,
            device=DEVICE,
            model_warmup=False,
            batch_size=16,
            rerank_listwise=True,
            rerank_passages_per_block=2,
        )
    )


@needs_download
def test_capabilities_and_score_range(engine):
    assert engine.capabilities == {"rerank"}
    assert engine.score_range == "cosine"
    assert engine._rerank_listwise is True
    assert engine.rerank_passages_per_block == 2
    # per-family reference caps (v3: 512/2048; v3.5 hard-codes 1024/8192)
    import os as _os

    if _os.getenv("JINA_RERANKER_V3_MODEL", "").endswith("v3.5"):
        assert (engine._max_query_length, engine._max_doc_length) == (1024, 8192)
    else:
        assert (engine._max_query_length, engine._max_doc_length) == (512, 2048)


@needs_download
def test_prompt_layout_uses_remote_format_verbatim(engine):
    prompt = engine._build_prompt(QUERY, DOCS[0])
    # the system block is stable across jina's active revisions
    assert prompt.startswith(
        "<|im_start|>system\nYou are a search relevance expert who can determine a ranking"
    )
    assert "<|im_start|>user\n" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert "Rank the passages based on their relevance to query:" in prompt
    # both readout tokens must be embedded in the prompt
    assert "<|embed_token|>" in prompt
    assert "<|rerank_token|>" in prompt
    # no_thinking=True: the prompt must end with the remote's own suffix
    # literal (bytecode const of the loaded function), taken verbatim.
    # Locate it dynamically so the test survives upstream format changes.
    own_suffixes = [
        c
        for c in engine._prompt_fn.__code__.co_consts
        if isinstance(c, str) and "think" in c and "assistant" not in c
    ]
    assert own_suffixes, "no_thinking suffix literal not found in remote function"
    assert any(prompt.endswith(c) for c in own_suffixes)

    ids = engine.encode_pre([(QUERY, DOCS[0])])["input_ids"][0].tolist()
    assert engine.model.doc_embed_token_id in ids
    assert engine.model.query_embed_token_id in ids


@needs_download
def test_pairwise_orders_relevant_first(engine):
    pairs = [(QUERY, d) for d in DOCS]
    scores = engine.encode_post(engine.encode_core(engine.encode_pre(pairs)))
    assert len(scores) == len(DOCS)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)
    order = np.argsort(scores)[::-1]
    ranked = [DOCS[i] for i in order]
    # Paris docs must rank above the German docs for this query
    assert "Berlin" not in ranked[0]
    assert ranked[0] in (DOCS[0], DOCS[2], DOCS[3])
    assert ranked[-1] in (DOCS[1], DOCS[4])


@needs_download
def test_pairwise_matches_remote_single_doc(engine):
    """A 1-doc Option A prompt is byte-identical to the remote rerank() prompt
    for a 1-doc call, so the cosine scores must agree."""
    for query, doc in [(QUERY, DOCS[0]), (QUERY, DOCS[2]), (QUERY, DOCS[1])]:
        ours = float(
            engine.encode_post(engine.encode_core(engine.encode_pre([(query, doc)])))[0]
        )
        remote = engine.model.rerank(query, [doc], top_n=1)[0]["relevance_score"]
        assert ours == pytest.approx(float(remote), rel=1e-4, abs=1e-4)


@needs_download
def test_listwise_matches_remote_with_matching_blocks(engine):
    """passages_per_block=125 reproduces the remote default block logic
    (same block composition → same weighted query embedding → same scores)."""
    order, scores = engine.rerank_list(QUERY, DOCS, passages_per_block=125, top_n=5)
    remote = engine.model.rerank(QUERY, DOCS, top_n=5)
    assert [r["index"] for r in remote] == order
    np.testing.assert_allclose(
        [r["relevance_score"] for r in remote], scores, rtol=1e-4, atol=1e-5
    )


@needs_download
def test_listwise_small_blocks_rank_correctly(engine):
    """With default 2-passage blocks the ranking must stay sane (paper's
    finding: the weighted query embedding is robust to block size)."""
    order, scores = engine.rerank_list(QUERY, DOCS, passages_per_block=2, top_n=5)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)
    ranked = [DOCS[i] for i in order]
    assert ranked[0] in (DOCS[0], DOCS[2], DOCS[3])
    # same top-1 as the remote default-block run
    top1_remote = engine.model.rerank(QUERY, DOCS, top_n=1)[0]["index"]
    assert order[0] == top1_remote
