"""End-to-end check of the TabbyAPI rerank endpoint against a golden record.

Assumes TabbyAPI is running with the Jina reranker v3 loaded as the
embeddings model. Verifies, per query:

- the ranking order matches the model's native listwise/pairwise ordering,
- normalized scores match the golden [0, 1] values within tolerance,
- raw_scores=True returns native cosine scores in [-1, 1].

Usage:

    .venv/bin/python docs/scripts/jina_reranker_e2e.py [--url http://127.0.0.1:8000] [--listwise]

The model must have been loaded with rerank_listwise set correspondingly
(listwise mode requires it).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

TOL = 1e-3

GOLDEN = json.loads(
    (Path(__file__).parent / "jina_reranker_golden.json").read_text()
)


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--listwise", action="store_true", help="expect listwise (Option B) behavior"
    )
    args = ap.parse_args()

    failures = 0
    for entry in GOLDEN["listwise" if args.listwise else "pairwise"]:
        query = entry["query"]
        payload = {"query": query, "documents": GOLDEN["documents"]}

        # ordered + normalized
        resp = post(args.url, "/v1/rerank", payload)
        got = [
            (r["index"], r["relevance_score"])
            for r in resp["results"]  # type: ignore[attr-defined]
        ]
        if args.listwise:
            expected_order = entry["order"]
            expected_norm = dict(
                zip(expected_order, entry["normalized_scores"])
            )
        else:
            expected_norm = dict(enumerate(entry["normalized_scores"]))
            expected_order = sorted(
                expected_norm, key=expected_norm.get, reverse=True
            )

        got_order = [idx for idx, _ in got]
        got_norm = dict(got)

        order_ok = got_order == expected_order
        score_ok = all(
            math.isclose(got_norm[str(i)], expected_norm[i], rel_tol=TOL, abs_tol=TOL)
            for i in expected_norm
        )

        # raw passthrough
        payload["raw_scores"] = True
        resp_raw = post(args.url, "/v1/rerank", payload)
        raw = {r["index"]: r["relevance_score"] for r in resp_raw["results"]}  # type: ignore[attr-defined]
        raw_ok = all(
            math.isclose(raw[str(i)], entry_expected, rel_tol=TOL, abs_tol=TOL)
            for i, entry_expected in enumerate(entry["raw_scores"])
        )

        status = "OK " if order_ok and score_ok and raw_ok else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"[{status}] q={query[:45]!r} "
            f"order_ok={order_ok} norm_ok={score_ok} raw_ok={raw_ok}"
        )

    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
