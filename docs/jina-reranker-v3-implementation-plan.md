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

**Wave 0 — DONE (2026-10-01); read `docs/jina-reranker-wave0-baseline.md`.**
Environment resolved and baseline captured: transformers 4.57.6 + ST 3.3.1 +
torch 2.14.1+cpu (+torchvision +cpu) + pytest 8.x; `infinity_emb/compat.py`
ST shim shipped on the infinity branch (CodeCarbonCallback re-exposure);
clone `pyproject.toml` numpy bound relaxed `<2`→`<3`. Baseline: 63 pass /
15 fail (all optimum/ct2/vision extra gaps or pre-existing CLI/server test
issues). W-A must treat these as already-shipped inputs, not re-derive them.

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
Mission: land the PR #674 borrow set on top of Wave 0: `requirements-uv.in` +
`requirements-uv.lock.txt` (uv-authoritative pin set mirroring the Wave-0
venv: transformers 4.57.6, tokenizers 0.22.2, ST 3.3.1, torch 2.14.1+cpu,
pytest 8.x, numpy 2.x); the BetterTransformer fallback fix (acceleration.py);
removal of Docker git-install lines; Qwen3 smoke tests + README model rows;
re-run and publish the baseline. **Already shipped by Wave 0 (verify only):**
`compat.py` + `__init__.py` hook, numpy bound relax, editable-install
viability.

Files (exclusive):
- `pyproject.toml` (line 33: `>=4.47.0` → `>=4.51.0`; keep `<=5.0`) — hand edit,
  do not restructure (stays poetry-layout for upstream compatibility)
- `requirements-uv.in` (new; direct pins per Wave-0 baseline: `transformers==4.57.6`,
  `tokenizers==0.22.2`, `sentence-transformers==3.3.1`, `huggingface_hub<1.0`,
  `torch==2.14.1+cpu` from the cpu index, `pytest>=8,<9`) +
  `requirements-uv.lock.txt` (new; `uv pip compile`) — the **authoritative**
  lock; upstream `poetry.lock` is left untouched
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
Wave 0  ✅ DONE (2026-10-01) — env + baseline recorded in
        docs/jina-reranker-wave0-baseline.md; commits on both feature branches
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

---

## Wave 1 — DONE (2026-10-01); executed as four contract-locked workstreams

All four workstreams landed and were verified **against the real model**
(`jinaai/jina-reranker-v3` @ `d7d7e73b…`, ~1.2 GB download, CPU).

**Commits:** infinity `69bca1f` · TabbyAPI `a6efdf7` (+ docs/scripts fixes).

**W-A (matrix/docs):** transformers floor `>=4.51`, dev pin `4.57.6`;
BetterTransformer import now catches `RuntimeError` (optimum ≤1.27) with a
module-level reason + warning; all 5 Dockerfiles drop the
`transformers@7547f55e` git build; `requirements-uv.in` +
`requirements-uv.lock.txt` (38 pins, `uv pip compile` with
`--index-strategy unsafe-best-match`); README rows (jina-reranker-v3/v3.5,
Qwen3-Embedding); gated Qwen3-Embedding ST smoke test.

**W-B (detection + Option A):** `select_model.py` maps exact arch
`JinaForRanking` → `RerankEngine.jina_v3` (takes precedence over seq-cls);
`crossencoder/jina_v3.py` loads via `AutoModel(trust_remote_code=True)`,
builds prompts with the remote module's `format_docs_prompts_func` verbatim
(resolved from the loaded module — resilient to upstream format drift),
relies on the remote `forward()` computing cosine scores from the readout
token positions; `score_range = "cosine"`.

**W-C (score semantics + Option B):** `BatchHandler.rerank` normalizes
range-aware (C2: sigmoid for logits engines, `0.5·(x+1)` for cosine); listwise
routing (`_rerank_listwise`) with `rerank_list()` (Option B) mirroring the
remote `rerank()` reference logic (block flush, per-block max-normalized
weights, request-global weighted query embedding), serialized by an
engine-level lock (C3); `EngineArgs` gains `rerank_listwise` /
`rerank_passages_per_block` (validated ≥1).

**W-D (TabbyAPI surface):** `InfinityContainer.load` threads both knobs into
`EngineArgs`; `EmbeddingsConfig` + `EmbeddingModelLoadRequest` + 
`config_sample.yml` document them; `/v1/rerank` handler notes cosine
raw-score semantics; e2e harness: `docs/scripts/generate_golden.py`,
`docs/scripts/jina_reranker_golden.json`, `docs/scripts/jina_reranker_e2e.py`.

### Wave-1 verification results (§6 gates)

| Gate | Result |
|---|---|
| G1 model loads | ✓ (pinned revision, fresh modules cache) |
| G2 prompt fidelity | ✓ verbatim — test asserts **bytecode-const** suffix of the loaded function, immune to display/upstream drift |
| G3 fidelity A vs remote | ✓ pairwise scores == remote `rerank()` single-doc, rel 1e-4 |
| G3 fidelity B vs remote | ✓ `passages_per_block=125` == remote block-for-block (rtol 1e-4) |
| G4 score contract | ✓ stub matrix (logits/cosine × raw/not) + live engine through `AsyncEmbeddingEngine` (order + scores vs golden, both modes) |
| G5 no regression | ✓ full unit suite: 69 pass / 21 fail — all out-of-matrix (optimum/ct2/diskcache/CLI env, pre-existing) |

### Findings folded back (ground truth)

1. **The remote prompt format is load-bearing and revision-locked.** The
   pinned revision renders `assistant\n<thinking>\n\n</thinking>\n\n` for
   `no_thinking=True`; upstream has changed this literal across pushes, so
   engines/tests must never hard-code it — always reuse the loaded module's
   function (authenticated by bytecode-const checks).
2. **`forward()` needs no manual readout math at batch level** — the remote
   code locates token ids 151670/151671 internally and returns `scores`.
3. **`transformer/utils.RerankEngine.jina_v3` + arch detection at
   config.json level** is enough; `spawn`/worker plumbing unchanged (Option A
   uses the existing per-pair pipeline; Option B routes by flag).
4. Golden scores are **keyed by document index** (not rank) to avoid
   compare-ambiguity in the harness.

### Open items (Wave 2, on request)

- v3.5 matrix check (needs `jinaai/jina-reranker-v3.5` weights + its
  truncation defaults query 1024/doc 8192 and `thinking` prompt variant);
- CLI flags for `--rerank-listwise` (`INFINITY_RERANK_LISTWISE`) — fields
  exist in `EngineArgs`, CLI auto-gen verified only for the API path;
- running `docs/scripts/jina_reranker_e2e.py` against a live TabbyAPI
  instance (requires the TabbyAPI runtime env; the harness + golden are CI-able
  as-is); the AsyncEmbeddingEngine seam was verified in-env.

---

## Wave 2 — GPU + v3.5 (DONE 2026-10-01)

GPU (RTX 3080 Laptop 16 GB, sm_86, driver 576.52, compute 8.6) became
available. Everything below verified on `device=cuda` with the built matrix.

**Environment hardening (uv only):**
- The dev venv's torch was `+cpu`; swapped to `torch==2.14.1+cu126` /
  `torchvision==0.29.1+cu126` (the only CUDA variant published for 2.14.1).
- The system `/usr/bin/python3.13` had **no `Python.h`** → Triton's first
  CUDA JIT compile crashed (`Python.h: No such file or directory`). Rebuilt
  the venv on a **uv-managed CPython 3.13.15** (ships headers); Triton now
  compiles its `cuda_utils` at first kernel launch.
- New `requirements-uv-gpu.in` + `requirements-uv-gpu.lock.txt` (pinned to
  `+cu126`, recompiled from the env each time); both locks recompile with
  `--index-strategy unsafe-best-match --index-url <pytorch index>
  --extra-index-url https://pypi.org/simple`. Install honors the same flags.
- Test tooling: the `pytest-anyio` PyPI distribution is a 0.0.0 stub on this
  index; the plugin ships inside `anyio` itself — pin `anyio>=4` in both
  locks.

**Engine hardening (`crossencoder/jina_v3.py`):**
- Truncation caps are per-family: reranker-v3 exposes `max_query_length`
  (512) / `max_doc_length` (2048) as `rerank()` signature defaults; **v3.5
  hard-codes 1024/8192 in the remote body**. `_resolve_truncation_caps()`
  now unwraps `@torch.no_grad()` via `__wrapped__`, reads signature defaults
  first, then bytecode constants; `rerank_list()` uses the resolved caps.
- New dtype warning: loading in bf16/fp16 drifts cosine scores ~1e-3 vs
  float32 (ordering unaffected). Use `dtype=float32` for golden-grade
  cross-device score reproducibility (measured: max diff 8.4e-5 vs 1.9e-3).

**TabbyAPI surface:** new `embeddings_dtype` config (auto|float32|float16|
bfloat16) threaded `config_sample.yml` → `EmbeddingsConfig` →
`EmbeddingModelLoadRequest` → `InfinityContainer.load` → `EngineArgs.dtype`.
Harness supports per-model goldens (`--golden`); v3.5 golden committed.

**Verification (CUDA):**
- v3 and v3.5 each pass the full model-backed suite (`JINA_DEVICE=cuda`,
  both suites: 6/6) — prompt-verbatim, pairwise == remote single-doc,
  listwise == remote block-for-block **using the resolved per-family caps**.
- AsyncEmbeddingEngine e2e on cuda: both modes order-exact vs golden;
  fp32 restores score-exact (8.4e-5), bf16 keeps order.
- Full unit suite re-run on the rebuilt env (see run log; only the
  documented out-of-matrix failures remain).

**Wave 3 — CLI flags, done:** `--rerank-listwise` / `INFINITY_RERANK_LISTWISE`
and `--rerank-passages-per-block` / `INFINITY_RERANK_PASSAGES_PER_BLOCK`
added to the v2 CLI + env manager and verified end-to-end on CUDA:
`infinity_emb v2 ... --rerank-listwise` serves `/rerank` with http 200;
pairwise vs listwise agree on ordering for a 3-doc request (Δ=0.068 in
cosine — expected: listwise query attends to all docs), each mode already
proven faithful to the reference by the fidelity suite. Dev matrix now
includes the `[server]` deps the CLI/server path needs (typer,
prometheus-fastapi-instrumentator, uvicorn[standard], orjson).

**Open (Wave 4):** live TabbyAPI server e2e run (the venv is the *infinity*
env — TabbyAPI's own runtime env still needs to be booted by the operator;
harness + both goldens are ready).

---

## Note — Jina's llama.cpp/GGUF requirements comment (2026)

jinaai/jina-reranker-v3.5-GGUF states the model "requires a non-causal encoder
mode and a custom --output-token-ids flag that are not yet in the official
llama.cpp release."

**Assessment: not applicable to this feature.** Those are llama.cpp serving
runtime gaps (GGUF has no encoder-mode pass / no token-position output
plumbing). Our engine uses the official HF `transformers` forward via the
model's own remote `modeling.py`, which extracts the readout-token hidden
states internally and returns `scores` (cosine) from `forward()` — no
token-level output hook is needed. Fidelity gates prove equivalence to the
reference `rerank()` (pairwise rel 1e-4; listwise block-for-block; both
models, CPU + CUDA). Watch-point: if Jina later changes `modeling.py` itself
(e.g., bidirectional attention to match llama.cpp), the engine follows the
loaded module automatically and the fidelity tests re-verify.
