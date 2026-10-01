# Feasibility Report: Jina reranker v3 family in the Infinity server

**Scope:** Assess whether `jinaai/jina-reranker-v3` and `jinaai/jina-reranker-v3.5`
can be served through the rerank path that TabbyAPI uses, i.e. the `infinity/`
clone (`libs/infinity_emb`, v0.0.77, commit `1eb4396`).
**No code was written.** All claims below were re-verified against the current
repo state and Hugging Face (2026-03).

---

## 1. Executive summary

| Model | Detection today | Pipeline fit today | Verdict |
|---|---|---|---|
| `jina-reranker-v3` / `v3.5` | ❌ fails | ❌ mismatch at every layer (loader, templating, scoring, batching) | **Low** – real feature work; a new engine class + BatchHandler decision, plus the LBNL constraints (listwise-only scoring, verbatim prompt, passages-per-block tuning) |

**Bottom line:** the v3 family is the LBNL listwise reranker (arXiv 2509.25085)
whose scoring math, listwise-only batching, and verbatim-prompt requirement are
fundamentally different from the current `logits[0] + sigmoid` pipeline. It is
*feasible* but is a multi-day feature, not a wiring fix; and it demands the
listwise path — pairwise feeding produces deterministic-but-meaningless scores.

---

## 2. Corrections to the findings doc

The prior findings (also authored for this project) need three corrections after
re-verification against the live model repos:

1. **Scoring is NOT logit extraction.** The shipped `modeling.py` for
   `jina-reranker-v3` and `jina-reranker-v3.5` is structurally identical: it
   appends `<|embed_token|>` (docs) / `<|rerank_token|>` (query) to a
   chat-template prompt, runs the qwen3 decoder, extracts hidden states at
   those tokens, projects to 512 dims, and scores with **cosine similarity**.
   `forward()` returns `scores`, with `logits=None`. There is no
   "logit over generated positions" step in the current remote code, and
   `use_cache=False` is forced.
2. **Size is ~0.6B, not 3.7B.** hidden 1024 × 28 layers, GQA (kv=8),
   intermediate 3072, vocab 151936 with tied embeddings → ≈12·h²·L ≈ 352M
   transformer params + ~156M shared embedding ≈ **0.5–0.6B**, matching the
   paper's "0.6B" (v3 and v3.5 are both this size). bf16 weights ≈ 1.2 GB —
   CPU-deployable, contrary to the 3.7B/7.4 GB assumption.
3. **No ONNX / no sentence-transformers export ships.** Neither model repo has
   an `onnx/` folder, `modules.json`, or GGUF (v3.5 has exactly 12 sibling
   files). The `optimum` backend and the SentenceTransformer loader are both
   dead ends for this family — the torch backend with a custom loader is the
   only path.

---

## 3. Architecture fact sheet — LBNL, per the paper

Primary source: *jina-reranker-v3: Last but Not Late Interaction for Listwise
Document [Re-]Ranking*, arXiv 2509.25085 (LBNL = **"Last But Not Late"**
interaction — the family's architectural principle). Diffed `modeling.py` of
`jina-reranker-v3` vs `jina-reranker-v3.5` — identical apart from truncation
defaults (v3: query 512 / doc 2K tokens; v3.5: query 1K / doc 8K).

- **Listwise, one shared 131K window.** `max_position_embeddings = 131072`
  (128K). One prompt holds query + all candidates, so Σ(docs)+query must fit
  the window — this is why truncation/batching exists at all.
- **The LBNL readout layout (verbatim from `format_docs_prompts_func`):**
  `system/<|im_start|>…` + `user` role + instruction line that embeds the query
  text, then each passage as `<passage id="i">\n{doc}<|embed_token|>\n</passage>`,
  then `<query>\n{query}<|rerank_token|>\n</query>`, then the assistant prefix
  `…<|im_start|>assistant\n thinking\n\n response\n\n`. The LM head is
  disabled (`self.lm_head = nn.Identity()`); `forward()` returns `scores` with
  `logits=None`. Hidden states at the readout tokens (doc `<|embed_token|>` id
  151670, query `<|rerank_token|>` id 151671) are projected by a 2-layer MLP to
  512-dim and scored by **cosine similarity** — i.e. relevance is read from each
  document's terminal token, exactly the paper's "last token" readout.
- **Which token carries the list view (causal-masking precision):** with causal
  attention, a doc's readout token attends only to *preceding* tokens (system,
  instruction-with-query, earlier passages) — it cannot see later documents.
  The full-list view lives in the **query embedding**, read at the final
  `<|rerank_token|>` which attends to everything including all passages.
  Cross-document comparison is therefore mediated through that global query
  readout: score(doc_i) = cos(projected doc_i embed, projected global query
  embed). Feeding one (query,doc) pair into a pointwise path puts the readout at
  a position/semantics the model was never trained to score → deterministic but
  not LBNL-meaningful results (matches the observed llama.cpp behavior).
- **Training distribution:** lists of up to ~16 passages per prompt (paper /
  v3 best-practice notes). The shipped `rerank()` hard-codes `block_size = 125`
  and flushes blocks by `model_max_length − 2·query_length` — beyond the
  calibrated range; an adapter should expose passages-per-block and prefer
  ~16 as the operating point, larger lists extrapolate.
- **The exact prompt is load-bearing.** Embeddings land at fixed token IDs, so
  the score only exists if the prompt is reproduced *verbatim* (roles, system
  text, thinking-prefix suffix, token order). Do **not** use
  `tokenizer.apply_chat_template` (different layout) — reuse
  `format_docs_prompts_func` from the loaded remote module, including
  `sanitize_input` (strips the special tokens from query/doc text so readout
  positions can't collide).
- v3 uses full attention (`use_sliding_window: false`); v3.5 mixes
  sliding/full layers (`sliding_window: 1024`, `use_cache: false`).
- Consequence for integration: per-pair prompts (Option A) are a degenerate but
  format-valid LBNL input (query + 1 passage) — scores are meaningful as
  single-doc relevances, but drop the cross-doc/global normalization the model
  applies; faithful scoring for a whole candidate set means one window per list
  (Option B in §6).

---

## 4. Verified model data (HF, 2026-03)

| Model | `architectures` | `model_type` | `max_position_embeddings` | remote code | ONNX | modules.json |
|---|---|---|---|---|---|---|
| `jina-reranker-v3` | `["JinaForRanking"]` | `qwen3` | 131072 | yes (`AutoModel` → `modeling.JinaForRanking`) | no | no |
| `jina-reranker-v3.5` | `["JinaForRanking"]` | `qwen3` | 131072 | yes (`AutoModel` → `modeling.JinaForRanking`) | no | no |

Configs checked at:
`jinaai/jina-reranker-v3/raw/main/config.json` and
`jinaai/jina-reranker-v3.5/raw/main/config.json`; `modeling.py` downloaded and
read in full for both (diffed — identical apart from truncation defaults).

---

## 5. How the pieces fit today (verified in the clone)

The rerank path through the installed `infinity-emb` (never the clone):

```
POST /v1/rerank                      TabbyAPI  endpoints/OAI/utils/rerank.py
  → InfinityContainer.rerank(...)    backends/infinity/model.py (EngineArgs built here)
    → AsyncEmbeddingEngine.rerank    infinity_emb/engine.py  (query, docs, raw_scores, top_n)
      → BatchHandler.rerank          inference/batch_handler.py
          sigmoid (unless raw_scores) → sort desc → top_n
      → ModelWorker pipeline         one item = one (query, document) pair
          encode_pre  = tokenize pair   (transformer/crossencoder/torch.py)
          encode_core = forward, take ["logits"]  → [N_pairs] logits
          encode_post = flatten to float list
```

Touch points, with the exact files:

| Concern | File in clone | Notes |
|---|---|---|
| **Detection** | `inference/select_model.py` → `get_engine_type_from_config()` | Only rule: `SequenceClassification` in arch + `len(id2label) < 2` → rerank. Single, well-isolated decision point. |
| **Dispatch** | `transformer/utils.py` → `RerankEngine` enum | Only `torch` and `optimum` entries. |
| **Loader** | `transformer/crossencoder/torch.py` → `CrossEncoderPatched` | Hard-wired to sentence-transformers `CrossEncoder` (which instantiates via `AutoModelForSequenceClassification`; no `model_class` override exposed). `trust_remote_code` is threaded and defaults **True** (`args.py`, from `env.py`). |
| **Templating** | `encode_pre` | Raw `(query, doc)` pairs, `truncation="longest_first"`. No chat-template hook. |
| **Scoring** | `encode_core` | Reads `out["logits"]`; `BatchHandler.rerank` then applies sigmoid. |
| **Batching** | `inference/batch_handler.py` | Items are individual pairs; one rerank request = N queue items. No request-group concept. |
| **Capabilities** | `transformer/abstract.py` | `BaseCrossEncoder.capabilities = {"rerank"}`; missing capability → `ModelNotDeployedError` → TabbyAPI HTTP 400 (works already). |
| **TabbyAPI surface** | `backends/infinity/model.py` | Builds `EngineArgs(model_name_or_path, engine="torch", device=<cpu default>, bettertransformer=False, model_warmup=False)`. |
| **Tests / docs** | `tests/unit_test/transformer/crossencoder/test_torch_crossencoder.py`, `tests/unit_test/test_engine.py::test_engine_reranker_torch_opt`, `README.md` §Reranking | Templates exist for parametrizing a new family; README model list does not cover the v3 family. |

Environment facts: the shell has Python 3.13.5 + `uv`, but **no torch /
transformers / sentence-transformers / infinity-emb installed**; TabbyAPI pins
`sentence-transformers < 4.0` (i.e. 3.x, which supports `trust_remote_code`),
`huggingface_hub < 1.0`, and gets `transformers >=4.47, <=5.0` via
infinity-emb. The v3.5 config targets `transformers_version 4.57.3` — inside
the allowed range, but remote-code compatibility is the top open question (§8).

---

## 6. Feasibility: the v3 family in the current pipeline

What the model actually is (from `modeling.py`, read in full):
- qwen3 decoder (**~0.6B**, 28 layers, hidden 1024; v3: full attention,
  v3.5: sliding window 1024) + a 2-layer projector (→512d) and special tokens
  `<|embed_token|>` / `<|rerank_token|>`. See the LBNL fact sheet, §3.
- One prompt holds the query **and all documents**; a single forward returns
  doc+query embeddings at the special tokens; scores =
  cosine(query_embed, doc_embed) ∈ [-1, 1]. `logits=None` in outputs.
- The remote code ships its own `rerank(query, documents)` doing **block
  batching** (~125 docs/prompt, fit to `max_length − 2·query_len`), per-block
  max-normalized weights, request-global weighted-average of query embeddings,
  then sort + `top_n` itself.

Why each of the four gaps is real:

1. **Detection:** `JinaForRanking` matches nothing → embedder branch → no
   `modules.json` → load fails. Needs an allowlist entry. (Match the arch name
   *exactly* — `"JinaForRanking" in architectures` — so similarly-named decoder
   rerankers aren't accidentally caught.)
2. **Loader:** `CrossEncoder` instantiates via `AutoModelForSequenceClassification`;
   the remote `auto_map` registers `JinaForRanking` under `AutoModel` only → the
   load raises regardless of `trust_remote_code`. A new engine class (direct
   `AutoModel.from_pretrained(...)` load) is required.
3. **Templating/scoring:** pair-tokenization + `["logits"]` + sigmoid does not
   apply. Scores come from cosine similarity of projected special-token hidden
   states; score semantics should be [-1,1]-aware (sigmoid is monotone so
   *order* survives, but `raw_scores` then means "cosine sim" rather than the
   endpoint's usual "logit" contract — an API-contract decision, §8 risk 3).
4. **Batching:** the intended, correct, and fast path is *one model call per
   block of docs*, with request-global normalization across blocks. Today's
   pipeline schedules one queue item per `(query, doc)` pair, so the integration
   must choose between:
   - **Option A – pairwise adapter (minimal):** new engine class with
     `encode_pre/encode_core/encode_post` over pairs, single-doc prompts (a
     degenerate but format-valid LBNL input). Fits the existing
     `BatchHandler`/`ModelWorker` unchanged (~2–4 days). Downside: the query is
     re-encoded once per document (at ~0.6B that is affordable, but still ~N×
     the cost of one listwise pass), and results are single-doc relevances that
     drop the cross-doc/global normalization (per-pair cosine vs weighted-global).
   - **Option B – request-grouped engine (correct/performant):** introduce a
     "query-group" item so one `/v1/rerank` request becomes one (or few) model
     calls honoring the model's block logic and global normalization. Requires
     touching `BatchHandler` scheduling (`_schedule`/`_get_prios_usage` or a new
     item type) — the deeper change (+2–3 days), but it is what makes v3/v3.5
     usable at all at scale and keeps scores faithful to the model's design.
   - A middle variant (BatchHandler unchanged, but the engine class internally
     accumulates all pairs of a request before calling the model once) fails on
     concurrency (queue items from different requests interleave) and
     batch-size limits — do not rely on it.

**Operational reality for TabbyAPI specifically:**
- TabbyAPI defaults `device="cpu"` (`embeddings_device: cpu`) and passes no
  dtype/quantization. At ~0.6B / bf16 (≈1.2 GB weights) the family **can** run
  on CPU, but the LBNL path re-attends the full prompt per block with
  `use_cache=False` — expect seconds per block of up to ~16–125 passages at
  8K-token docs on CPU; a CUDA device is still the practical recommendation.
  Ideally thread more knobs through `InfinityContainer` (dtype, batch size,
  passages-per-block; see §3 training-distribution note). There is no
  `onnx`/`optimum` export for the family, so the torch backend is the only path;
  a CT2 or bitsandbytes quantization route would be new work upstream in
  infinity-emb anyway.

---

## 7. Recommended approach (phased, no code yet)

1. **Phase 0 – environment + repro (0.5–1 day).** Install
   `pip install -e ./infinity/libs/infinity_emb` (or match TabbyAPI's pins:
   `sentence-transformers < 4.0`, `huggingface_hub < 1.0`) + torch. Reproduce
   the current failure for `jina-reranker-v3.5` (`create_server(EngineArgs(
   model_name_or_path="jinaai/jina-reranker-v3.5", ...))` → `POST /rerank`):
   confirm the embedder-branch misrouting and the
   `AutoModelForSequenceClassification` load failure. Also verify the remote
   `modeling.py` loads cleanly under the TabbyAPI transformers pin.
2. **Phase 1 – detection (small).** Extend `get_engine_type_from_config` with
   an exact arch allowlist (`"JinaForRanking" in architectures` → rerank
   engine). Unit tests in `tests/unit_test/inference/test_select_model.py`.
3. **Phase 2 – v3/v3.5 engine (the real work).** New engine class (Option A
   first as a thin vertical slice; Option B when production quality is wanted),
   score-range-aware `raw_scores` handling (do not sigmoid cosine scores),
   exec through `BatchHandler`. Add parametrized tests modeled on
   `test_torch_crossencoder.py` / `test_engine_reranker_torch_opt`; validate
   scores against the model's own `rerank()` as ground truth.
4. **Phase 3 – TabbyAPI surface (small).** No changes needed for detection or
   routing. Thread a couple of `EngineArgs` knobs (dtype, batch size,
   passages-per-block) through `backends/infinity/model.py` and document
   `embeddings_device` guidance. Refresh README model lists.

Estimated total: **4–7 days** including tests, dominated by the
batching/score-semantics decision (Option A vs B) and remote-code validation.

---

## 8. Top risks & open questions

1. **Remote-code version skew.** The v3/v3.5 remote code pins `transformers`
   4.55–4.57; TabbyAPI allows 4.47–5.0. Success is empirical; maintain a
   known-good pinned matrix in CI. Jina remote code is unversioned (`main`) —
   behavior can drift under a deployed server.
2. **Listwise semantics are the hard constraint.** "Scores only mean what LBNL
   expects when the whole list shares the window" (§3) — this argues for the
   listwise Option B path and rules out pointwise feeding (deterministic but
   meaningless results).
3. **Score semantics across the endpoint.** The v3 family outputs cosine
   similarities ∈ [-1, 1]; the existing rerank endpoint sigmoid-transforms
   logits. Ordering is preserved (sigmoid is monotonic), but the meaning of
   `raw_scores` and the Cohere [0,1] relevance convention need an explicit
   decision + tests.
4. **Batching fidelity (Option A vs B).** Per-pair scoring is not equivalent to
   the model's block-normalized global scoring; A is a pragmatic approximation,
   B is faithful. Choose deliberately.
5. **Concurrency interplay** if Option B is chosen: request-group items must
   still respect `max_batch_size`, `batch_delay`, and overload accounting
   without letting concurrent requests corrupt grouping.
6. **CPU throughput.** At ~0.6B bf16, CPU (TabbyAPI default) is feasible but
   slow: full-prompt re-attention per block (`use_cache=False`). GPU guidance
   for operators is required.

---

## 9. Files that would change (map for the implementation session)

| File | Change |
|---|---|
| `infinity/libs/infinity_emb/infinity_emb/inference/select_model.py` | detection rules / allowlist |
| `infinity/libs/infinity_emb/infinity_emb/transformer/utils.py` | possibly a new engine enum entry |
| `infinity/libs/infinity_emb/infinity_emb/transformer/crossencoder/torch.py` (or new `crossencoder/jina_v3.py`) | loader + templating + scoring for v3 |
| `infinity/libs/infinity_emb/infinity_emb/inference/batch_handler.py` | (Option B only) request-group scheduling; score-range-aware sigmoid |
| `infinity/libs/infinity_emb/infinity_emb/engine.py` | docstrings/contract only |
| TabbyAPI `backends/infinity/model.py` | only if new `EngineArgs` knobs are needed |
| `libs/infinity_emb/tests/unit_test/transformer/crossencoder/…`, `tests/unit_test/inference/test_select_model.py`, `tests/unit_test/test_engine.py` | parametrized v3-family tests |
| `README.md` (infinity) + TabbyAPI docs | model-support lists |
