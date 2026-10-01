"""Unit tests for rerank score semantics and listwise routing in BatchHandler.

No model weights are involved: stub engines emulate the BaseCrossEncoder
contract (capabilities, encode_pre/core/post, tokenize_lengths). Covers
contract C2 (score-range aware normalization) and the Option B request
routing introduced for the JinaForRanking engine family.
"""

import threading
import time

import numpy as np
import pytest

from infinity_emb.inference import BatchHandler
from infinity_emb.transformer.crossencoder.jina_v3 import JinaV3CrossEncoder


class StubCrossEncoder:
    """minimal rerank model; native per-doc scores 1.0, 0.7, 0.4."""

    capabilities = {"rerank"}
    score_range = "logits"
    NATIVE = np.array([1.0, 0.7, 0.4])

    def encode_pre(self, input_tuples):
        return input_tuples

    def encode_core(self, features):
        return features

    def encode_post(self, features):
        return StubCrossEncoder.NATIVE[: len(features)].copy()

    def tokenize_lengths(self, sentences):
        return [len(s.split()) for s in sentences]


class StubCosineCrossEncoder(StubCrossEncoder):
    """JinaForRanking-like engine: cosine scores, never sigmoided."""

    score_range = "cosine"


class StubListwiseCrossEncoder(StubCrossEncoder):
    """JinaForRanking-like engine with the listwise path enabled."""

    score_range = "cosine"
    _rerank_listwise = True
    rerank_passages_per_block = 2
    calls: list = []

    def rerank_list(self, query, documents, passages_per_block=16, top_n=128, **kwargs):
        self.calls.append((query, list(documents), passages_per_block, top_n))
        return [2, 0, 1], [0.9, 0.7, 0.5]


async def _rerank(model, query, docs, raw_scores, top_n=None):
    bh = BatchHandler(model_replicas=[model], max_batch_size=32, batch_delay=0.01)
    await bh.spawn()
    try:
        return await bh.rerank(query, docs, raw_scores=raw_scores, top_n=top_n)
    finally:
        await bh.shutdown()


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("score_range", "raw_scores", "first_expected"),
    [
        ("logits", True, 1.0),  # pass-through
        ("logits", False, _sigmoid(1.0)),  # sigmoid
        ("cosine", True, 1.0),  # pass-through
        ("cosine", False, 1.0),  # linear remap 0.5*(x+1) — NOT sigmoid
    ],
)
async def test_rerank_score_semantics_matrix(score_range, raw_scores, first_expected):
    model = StubCosineCrossEncoder() if score_range == "cosine" else StubCrossEncoder()
    docs = ["paris capital of france", "berlin capital of germany", "munich bavaria"]
    results, usage = await _rerank(model, "where is france", docs, raw_scores)

    assert results[0].document == docs[0]
    assert results[0].relevance_score == pytest.approx(first_expected, abs=1e-6)
    assert results[1].relevance_score < results[0].relevance_score
    assert results[2].relevance_score < results[1].relevance_score
    if not raw_scores:
        assert results[0].relevance_score <= 1.0
    assert usage > 0

    # indices must survive sorting
    assert [r.index for r in results] == [0, 1, 2]


@pytest.mark.anyio
async def test_rerank_listwise_routes_to_engine():
    model = StubListwiseCrossEncoder()
    docs = ["d0 jina", "d1 jina", "d2 jina"]
    results, usage = await _rerank(model, "query jina", docs, raw_scores=False)

    # engine returned order [2, 0, 1] with cosine scores [0.9, 0.7, 0.5]
    assert [r.document for r in results] == ["d2 jina", "d0 jina", "d1 jina"]
    assert [r.relevance_score for r in results] == pytest.approx([0.95, 0.85, 0.75])
    assert [r.index for r in results] == [2, 0, 1]
    assert usage > 0
    assert model.calls[-1][2] == 2  # passages_per_block threaded through


@pytest.mark.anyio
async def test_rerank_listwise_raw_passthrough():
    model = StubListwiseCrossEncoder()
    docs = ["d0 jina", "d1 jina", "d2 jina"]
    results, usage = await _rerank(model, "query jina", docs, raw_scores=True)
    assert [r.relevance_score for r in results] == pytest.approx([0.9, 0.7, 0.5])


def test_listwise_requires_explicit_flag():
    engine = JinaV3CrossEncoder.__new__(JinaV3CrossEncoder)
    engine._rerank_listwise = False
    engine._listwise_lock = threading.Lock()
    engine._max_query_length, engine._max_doc_length = 512, 2048
    with pytest.raises(RuntimeError, match="rerank_listwise"):
        engine.rerank_list("q", ["d"], passages_per_block=2, top_n=1)


def test_listwise_concurrency_is_serialized():
    """The engine-level lock must prevent interleaved listwise forwards:
    the CPU model is shared across worker threads (contract C3)."""
    engine = JinaV3CrossEncoder.__new__(JinaV3CrossEncoder)
    engine._rerank_listwise = True
    engine._listwise_lock = threading.Lock()
    engine._max_query_length, engine._max_doc_length = 512, 2048
    intervals: list[tuple[float, float]] = []

    def fake_unlocked(*args, **kwargs):
        t0 = time.perf_counter()
        time.sleep(0.1)
        intervals.append((t0, time.perf_counter()))
        return [0], [1.0]

    engine._rerank_list_unlocked = fake_unlocked
    barrier = threading.Barrier(4)

    def run():
        barrier.wait()
        engine.rerank_list("q", ["d"], passages_per_block=2, top_n=1)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(intervals) == 4
    for (a0, b0), (a1, _b1) in zip(sorted(intervals), sorted(intervals)[1:]):
        assert a1 >= b0 - 1e-6, "listwise calls overlapped — lock is not serializing"


def test_rerank_passages_per_block_validation():
    from infinity_emb.args import EngineArgs

    assert EngineArgs(rerank_passages_per_block=0).rerank_passages_per_block == 16
    assert EngineArgs(rerank_passages_per_block=4).rerank_passages_per_block == 4
