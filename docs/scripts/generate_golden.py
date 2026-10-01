"""Generate golden rerank scores for JinaForRanking models (reranker v3/v3.5).

Produces docs/scripts/jina_reranker_golden.json used by the e2e harness
(jina_reranker_e2e.py) to verify a running TabbyAPI+infinity stack against
the model's native behavior.

Usage (from the TabbyAPI root, with the Wave-0 venv active):

    .venv/bin/python docs/scripts/generate_golden.py

The first run downloads ~1.2 GB of weights (jinaai/jina-reranker-v3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device
from infinity_emb.transformer.crossencoder.jina_v3 import JinaV3CrossEncoder

MODEL_ID = os.getenv("JINA_RERANKER_V3_MODEL", "jinaai/jina-reranker-v3")
REVISION = os.getenv(
    "JINA_RERANKER_V3_REVISION", "d7d7e73b6ea138ced340b83865931b5dfb6c97aa"
)

QUERIES = [
    "What is the capital of France?",
    "How do neural rerankers work?",
    "What is the tallest building in the world?",
]

DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "Neural rerankers use cross-encoders to score query-document pairs jointly.",
    "The Jina reranker-v3 uses a listwise prompt with readout tokens.",
    "Burj Khalifa in Dubai is the world's tallest building at 828 meters.",
    "France is a country in Western Europe; its capital is Paris.",
]


def main() -> None:
    engine = JinaV3CrossEncoder(
        engine_args=EngineArgs(
            model_name_or_path=MODEL_ID,
            revision=REVISION,
            device=Device.cpu,
            model_warmup=False,
            batch_size=16,
            rerank_listwise=True,
            rerank_passages_per_block=16,
        )
    )

    golden: dict = {
        "model": MODEL_ID,
        "revision": REVISION,
        "queries": QUERIES,
        "documents": DOCUMENTS,
        "pairwise": [],
        "listwise": [],
    }

    # Pairwise (Option A): per (query, doc) cosine scores; normalized [0, 1]
    for query in QUERIES:
        pairs = [(query, d) for d in DOCUMENTS]
        scores = engine.encode_post(engine.encode_core(engine.encode_pre(pairs)))
        raw = [float(s) for s in scores]
        golden["pairwise"].append(
            {
                "query": query,
                "raw_scores": raw,
                "normalized_scores": [0.5 * (s + 1.0) for s in raw],
            }
        )

    # Listwise (Option B): request-grouped block logic, engine-level.
    # Scores are keyed by DOCUMENT INDEX (input order), not rank position.
    for query in QUERIES:
        order, scores = engine.rerank_list(
            query, DOCUMENTS, passages_per_block=16, top_n=len(DOCUMENTS)
        )
        by_doc = {str(i): 0.0 for i in range(len(DOCUMENTS))}
        for rank, doc_idx in enumerate(order):
            by_doc[str(doc_idx)] = float(scores[rank])
        golden["listwise"].append(
            {
                "query": query,
                "order": order,
                "raw_scores_by_doc": by_doc,
                "normalized_by_doc": {
                    k: 0.5 * (v + 1.0) for k, v in by_doc.items()
                },
            }
        )

    out = Path(__file__).parent / "jina_reranker_golden.json"
    out.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"golden written to {out} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
