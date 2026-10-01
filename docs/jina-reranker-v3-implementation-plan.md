# Implementation Plan — Jina reranker v3 / v3.5 in TabbyAPI's vendored infinity-emb

**Owner/coordinator:** general coordinator (this role)
**Executors:** up to 4 parallel twin subagents (A–D), file-ownership-isolated
**Inputs:** `docs/jina-reranker-family-feasibility.md` (architecture & gaps),
PR #674 borrow list (transformers bump, BetterTransformer fix, Qwen3 smoke
tests, Dockerfile cleanup), verified model facts (HF, 2026-03/08).
**Deliverable of this document:** delegation matrix, interface contracts,
sequencing, gates, and the coordinator's verification protocol.

**Package management (project convention — mandatory):** uv ONLY. Every
environment create, install, and dependency-resolution step in this plan uses
`uv` (`uv venv`, `uv pip install`, `uv pip compile`, `uv run`). Raw `pip`,
`pipx`, and `poetry install/run` are banned for this work. The vendored
clone's `poetry.lock` is upstream state and is *not* consumed by our
environments; a uv-derived lock file is authoritative. Each subagent must
report the exact uv commands it ran and must not modify files outside its
ownership list.

---

## 1. Objectives and non-goals

**Goal:** a TabbyAPI operator can configure `jinaai/jina-reranker-v3` or
`jinaai/jina-reranker-v3.5` and call `POST /v1/rerank` with correct,
LBNL-meaningful scores, without relying on upstream infinity merging anything.

Non-goals (explicit scope cuts):
- No ONNX / optimum backend for this family (no export exists; torch only).
- No BetterTransformer enablement (stays `False`; sdpa is correct for the
  sliding-window layers of v3.5).
- No changes to embedding/classify/image/audio paths beyond what the
  transformers bump forces.
- Option B (listwise request-group batching) is a prototype gated by score
  fidelity; Option A (pairwise) is the guaranteed-shippable vertical slice.

## 2. Target architecture (post-change call flow)

```
POST /v1/rerank (TabbyAPI, endpoints/OAI/utils/rerank.py: raw_scores/top_n passthrough — unchanged)
  → InfinityContainer.rerank            backends/infinity/model.py (EngineArgs gains optional knobs, W-D)
    → AsyncEmbeddingEngine.rerank       infinity_emb/engine.py (contract docstrings only)
      ├─ Option A: BatchHandler.rerank (score-range-aware sigmoid, W-C) → per-pair items
      │    └─ JinaV3CrossEncoder     transformer/crossencoder/jina_v3.py  (W-B)
      │         AutoModel.from_pretrained(trust_remote_code) → format_docs_prompts_func (verbatim)
      │         → readout tokens (<|embed_token|> 151670, <|rerank_token|> 151671)
      │         → cosine scores ∈ [-1,1]
      └─ Option B (prototype): BatchHandler request-group items
           └─ JinaV3ListwiseEngine   transformer/crossencoder/jina_v3_listwise.py (W-C)
                block batching (~16/passages_per_block), per-block max-normalization,
                request-global query-embedding weighted average → sort → top_n
Detection: inference/select_model.py — exact arch allowlist "JinaForRanking" → rerank (W-B)
```

## 3. Interface contracts (twin-safety: fixed now, never changed during the wave)

These are the seams that let agents B/C/D work in parallel without touching
each other's files.

**C1 — engine class contract** (`transformer/crossencoder/jina_v3.py`, owned by B):
```python
class JinaV3CrossEncoder(BaseCrossEncoder):
    capabilities = {"rerank"}
    score_range: Literal["logits", "cosine"] = "cosine"      # C2
    def __init__(self, *, engine_args: EngineArgs): ...      # AutoModel + AutoTokenizer, trust_remote_code=True,
                                                             # revision passthrough; use_cache=False (remote forces)
    def encode_pre(self, items) -> dict                      # per items: verbatim format_docs_prompts_func prompt
                                                             # (reused from the *loaded remote module*, NOT
                                                             # tokenizer.apply_chat_template) + sanitize_input; record
                                                             # readout-token indices
    def encode_core(self, features) -> list[float]           # forward → cosine per item ∈ [-1,1]; logits=None (never read)
    def encode_post(self, scores) -> list[float]
```
Registration: add `jina_v3 = JinaV3CrossEncoder` to `RerankEngine` in
`transformer/utils.py` (B). `select_model.py`: `"JinaForRanking" in
architectures` → `RerankEngine.jina_v3` (exact match, before the generic
SequenceClassification rule) (B). Do **not** dispatch JinaForRanking through
the sentence-transformers CrossEncoder loader — AutoModelForSequenceClassification
cannot resolve the `auto_map: {"AutoModel": ...}` entry.

**C2 — score contract** (implemented in `batch_handler.py` by C; consumed by
D's docs and the e2e harness):
| engine `score_range` | `raw_scores=True` → | `raw_scores=False` → |
|---|---|---|
| `"logits"` (existing) | raw logit | `sigmoid(logit)` (unchanged behavior) |
| `"cosine"` (new) | `cos` ∈ [-1,1] | `0.5*(cos+1)` ∈ [0,1] (neutral=0.5, ranking identical) |

BatchHandler reads the engine's `score_range`, never hard-codes sigmoid.

**C3 — Option-B request-group item** (C): one `/v1/rerank` request = **one**
queue payload `(query, docs, request_id, passages_per_block)`; the worker
chunks into blocks of `passages_per_block` (default **16** — training
calibrated, see report §3) sized against `max_batch_size` token limits; groups
never mix requests; per-block max-normalization and request-global query-embed
weighted average mirror the remote `rerank()` implementation. Availability is
behind `rerank_listwise=False` default (C4).

**C4 — EngineArgs knobs** (D, defaults fully backward-compatible):
`rerank_listwise: bool = False`, `rerank_passages_per_block: int = 16`,
`rerank_max_length: int | None = None` (guard for the 131072 window), plus
thread existing `dtype`/`batch_size` fields from TabbyAPI config. TabbyAPI
config keys: `rerank_listwise`, `rerank_passages_per_block`,
`embeddings_dtype`, with `embeddings_device` already supported.

## 4. Workstreams and subagent briefs

Each brief = mission / files (exclusive ownership) / contract obligations /
acceptance / constraints.

### W-A — Platform & dependencies (Agent A) — foundation, mostly standalone
Mission: land the PR #674 borrow set: `transformers >=4.51.0` (target
4.57.6), `tokenizers 0.22.2`; port the BetterTransformer fallback fix; remove
Docker git-install lines; add Qwen3 smoke tests + README model rows;
re-baseline the existing unit suite. All resolution work via uv.

Files (exclusive):
- `pyproject.toml` (line 33: `>=4.47.0` → `>=4.51.0`; keep `<=5.0`) — hand edit,
  do not restructure (stays poetry-layout for upstream compatibility)
- `requirements-uv.in` (new; direct pins: `transformers>=4.51,<=5.0`,
  `tokenizers==0.22.2`, `sentence-transformers>=3.0,<4.0`, `huggingface_hub<1.0`,
  `torch>=2.9`) + `requirements-uv.lock.txt` (new; generated with
  `uv pip compile requirements-uv.in -o requirements-uv.lock.txt`) — the
  **authoritative** lock; upstream `poetry.lock` is left untouched
- `infinity_emb/transformer/acceleration.py` (port PR hunk: module-level
  reason, `RuntimeError` catch, warning in `check_if_bettertransformer_possible`)
- `Dockerfile.jinja2` + generated `Dockerfile.*_auto` (delete the
  `pip install git+...@7547f55e` line)
- `tests/unit_test/transformer/embedder/test_torch.py` (port
  `test_sentence_transformer_qwen3_embedding`)
- new: `tests/unit_test/transformer/crossencoder/test_qwen3_seqcls.py`
  (loads `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` CPU, asserts it reranks via
  the existing torch crossencoder path — evidence for the seq-cls pivot)
- `README.md` (root, infinity) rows: Qwen3-Embedding-0.6B/4B/8B + `-seq-cls` reranker

Acceptance: `pytest tests/unit_test` green with **no pre-existing test made
flaky by the bump** (diff baseline vs post-bump; report any env-only failures
on Python 3.13 separately); resolved graph contains transformers ==4.57.x;
Qwen3 smoke matches the model-card similarity matrix.

Constraints: do not touch `select_model.py`, `batch_handler.py`, engine
files, or any TabbyAPI file. Pin every HF model used in tests to a revision.

### W-B — Detection + Option A pairwise engine (Agent B)
Mission: the Jina family loads and scores as documented. Purpose is a thin
vertical slice over the existing BatchHandler with **zero batch-handler
changes** (Option A).

Files (exclusive):
- `infinity_emb/inference/select_model.py` (arch allowlist, exact match)
- `infinity_emb/transformer/utils.py` (enum entry only)
- `infinity_emb/transformer/crossencoder/jina_v3.py` (new, contract C1)
- tests: `tests/unit_test/inference/test_select_model.py`,
  `tests/unit_test/transformer/crossencoder/test_jina_v3.py` (new,
  parametrized after `test_torch_crossencoder.py`; model-backed tests gated
  by `JINA_DOWNLOAD_TESTS=1` env var, pin revision)

Acceptance: detection tests pass; engine loads `jinaai/jina-reranker-v3` @
pinned revision on CPU; encode path returns cosine ∈ [-1,1]; on a fixed
3-query × 10-doc fixture, ordering equals the model's own `rerank()` output;
tokenization reuses the remote `format_docs_prompts_func` verbatim (test
asserts prompt equality with a golden rendered string, catching drift).

Constraints: own `jina_v3.py` completely; no changes to `batch_handler.py`
or `engine.py`; the Option-B method must be added by C in a **separate file**.

### W-C — Score semantics + Option B listwise prototype (Agent C)
Mission: implement C2 (score-range-aware `BatchHandler.rerank`) and the
listwise request-group prototype (C3), and prove Option B beats Option A on
fidelity (decision input for Gate G3).

Files (exclusive):
- `infinity_emb/inference/batch_handler.py` (score branch + Option-B item
  scheduling; must respect `max_batch_size`, `batch_delay`)
- `infinity_emb/engine.py` (docstrings/contract only)
- `infinity_emb/transformer/crossencoder/jina_v3_listwise.py` (new; imports
  B's engine for the decoder; owns block logic + global normalization)
- tests: `tests/unit_test/inference/test_batch_handler.py` (extend),
  `tests/unit_test/test_engine.py` (extend), new
  `tests/unit_test/transformer/crossencoder/test_jina_v3_listwise.py`

Acceptance: unit matrix for C2 (logits vs cosine × raw vs not); concurrency
test: two interleaved rerank requests return exactly their own results; with
`rerank_listwise=True`, benchmark results match remote `rerank()` (nDCG@10
≥ 0.95 or Spearman ≥ 0.99); C3 never exceeds `max_batch_size` token budget.

Constraints: no changes to `select_model.py`, `utils.py`, or B's
`jina_v3.py`; read B's class only via the public contract C1 surface.

### W-D — TabbyAPI surface + e2e harness (Agent D)
Mission: make the feature operable from TabbyAPI config; build the
verification harness and golden files; keep existing rerankers green.

Files (exclusive):
- `backends/infinity/model.py` (thread C4 knobs through `EngineArgs`)
- `config_sample.yml` + TabbyAPI docs (new keys, `embeddings_device`
  guidance: GPU recommended, CPU feasible ~0.6B/1.2 GB bf16)
- `endpoints/OAI/utils/rerank.py` (docstring only: raw_scores meaning is
  cos vs logit per model — no behavior change)
- `docs/fixtures/jina_rerank_benchmark.json` (generator script + committed
  20 queries × 100 docs, multilingual/EN/DE/ZH/code; golden top-10 IDs from
  the model's own `rerank()`)
- `tests/e2e/` (infinity) or TabbyAPI-side `scripts/e2e_jina_rerank.py`:
  spawn server with v3 (pinned revision) → POST /v1/rerank → assert 200,
  top_n respected, scores sorted, nDCG@10 vs golden ≥ threshold; regression
  rows: `bge-reranker-large`, `jinaai/jina-reranker-v1-turbo-en`

Acceptance (wave 2): harness executes green against B+C's real engine; bge
and jina-v1 rerank e2e still pass after all changes; config keys documented.

Constraints: never modify clone internals (A/B/C's files); depend on B/C
models only through the public contract; all model downloads pinned.

## 5. Sequencing, waves, gates

```
Wave 0  (coordinator + A, ~0.5-1 d)  env bootstrap + baseline (uv only)
   cd /home/dev/tabbyapi
   uv venv .venv --python 3.13
   uv pip install -e "infinity/libs/infinity_emb[torch]" \
       "sentence-transformers<4.0" "huggingface_hub<1.0"
   uv pip install "torch>=2.9" \
       --index-url https://download.pytorch.org/whl/cpu      # CPU dev wheels (py3.13)
   uv pip check
   cd infinity/libs/infinity_emb && uv run pytest tests/unit_test -q   # baseline record
Wave 1  (A ∥ B ∥ C ∥ D, 3-5 d)        parallel, contract-locked
Gate G1 (A): new lock green + Qwen3 smokes → unblocks B/C/D integration
Wave 2  (coordinator + B/C/D, 2-3 d)  integrate; generate golden; run e2e
Gate G2 (B): engine loads, /rerank answers, ordering matches on fixture
Gate G3 (fidelity, B↔C): Option A vs Option B vs remote rerank on the
   benchmark → nDCG@10(A) ≥ 0.95 ? ship Option A (C's listwise stays
   prototype) : require Option B (C fixes to gate, max 2 iterations)
Gate G4 (final): W-A..D suites green, ruff+mypy clean, docs complete,
   TabbyAPI manual curl on real model
Wave 3  (1-2 d)  perf/report: CPU vs GPU latency for 8K-token docs,
   memory notes, README rows final, release-note entry, option CI wiring
   for HF-gated tests
```

Total: ~7-11 days of coordinator-tracked work (feasibility report predicted
4-7 days for the engine alone; this adds the dependency foundation and
full validation harness).

## 6. Coordinator verification protocol (run at every gate)

1. **Ownership audit:** `git -C infinity diff --stat` — hunks outside each
   agent's allowed file list are rejected (use per-workstream allow-lists).
2. **Suites:** `pytest tests/unit_test -q` (clone), plus the e2e harness
   (D). Record which tests are HF-download-gated and run them once with
   `JINA_DOWNLOAD_TESTS=1` in the integration wave.
3. **Lint/types:** clone `make lint` (ruff+mypy per repo), TabbyAPI ruff.
4. **Deps:** resolved graph contains transformers 4.57.x, tokenizers 0.22.2,
   sentence-transformers 3.3.x, huggingface_hub <1.0; no optimum/colpali
   constraint conflict is permitted to touch the torch-only install.
5. **Numeric (G3):** `compare_scores.py` prints per-model nDCG@10,
   Spearman, and top-10-id Jaccard vs golden; thresholds from §4.
6. **No-regression:** e2e rows for bge-reranker-large and jina-v1-turbo
   must be green after every integration step.

## 7. Ground-truth validation spec

Golden source of truth = the model's own remote-code `rerank(query,
documents)` run at *planning time* in a reference env (transformers 4.57.6,
torch 2.8/2.9, pinned HF revisions — v3 `d7d7e73…`, v3.5 sha pinned the same
way). Everything is asserted against that:
- Option A/B ordering vs golden (nDCG@10 ≥ 0.95, Spearman ≥ 0.99);
- prompt-string equality vs golden rendered prompt (drift detector);
- readout positions: doc `<|embed_token|>` and query `<|rerank_token|>`
  token-ids resolved dynamically from the loaded tokenizer, asserted against
  151670/151671 known-good values.

## 8. Risks & mitigations (delta vs report §8)

| Risk | Mitigation |
|---|---|
| Remote-code drift (Jina `main` unversioned) | pin `revision` everywhere (tests, README, e2e); optional nightly sha-drift check |
| transformers 4.57 breaks Jina's `modeling.py` import of qwen3 internals | 4.57.3 declared target is inside 4.57.6 lock; if load fails, escalate to coordinator with traceback (single known-issue path, bus-factor 1) |
| Resolver conflicts (optimum-onnx `<4.58`, colpali `<4.58`, ST `<4.0`) | torch-only install; lock exact; optimum/vision extras documented as incompatible with the bump until re-locked (accept: A keeps them at old floor, noted in README) |
| CPU latency at 8K-token docs (full re-attention, `use_cache=False`) | benchmark at Wave 3; guidance: CUDA recommended, `passages_per_block` tunable; no shortcut exists (no ONNX export) |
| 131K window memory | `rerank_max_length` guard; mind RAM for multi-block requests |
| Option-B concurrency (group corruption) | unit test with interleaved requests; per-request_id isolation in C3 |
| HF downloads in CI | env-gated tests + committed golden files; no network in unit runs |
| `trust_remote_code` exposure | document policy; code comes from pinned revisions only |

## 9. Definition of done

1. `jinaai/jina-reranker-v3` and `jinaai/jina-reranker-v3.5` answer
   `POST /v1/rerank` with correct ordering; scores match golden per §7.
2. G3 decision recorded (Option A shipped / Option B required) with the
   fidelity table.
3. Existing rerankers (bge-reranker-large, jina-reranker-v1-turbo-en, plus
   the seq-cls Qwen3 reranker from #674) unaffected (e2e green).
4. TabbyAPI config keys (C4) documented in `config_sample.yml` + docs;
   README model lists updated.
5. Full unit + e2e suites green; lint/types clean; lock file committed;
   release-note entry added.

## 10. Handoff protocol (per subagent, returned at wave end)

- Summary of changes with rationale; test commands + full outputs;
- Deviations from contracts C1–C4, if any, with justification;
- Open risks / assumptions; anything needing the coordinator.
Coordinator runs §6 verification before accepting; rejected work returns
with the failing check attached.
