# Jina Reranker v3/v3.5 — Operating FAQ

Answers to the questions that came up while bringing the Jina reranker family
(jina-reranker-v3 / jina-reranker-v3.5) up on the `feat/jina-reranker-v3`
branch, via the vendored infinity-emb fork at `infinity/libs/infinity_emb`.

Companion doc: [jina-reranker-v3-implementation-plan.md](jina-reranker-v3-implementation-plan.md)

---

## Q1. I heard something about "chunking" documents — what does the engine actually do?

There are two different things, and only one of them is real chunking:

1. **Long documents → truncated, not chunked.** The model has per-family
   **token budgets** (from its own `_truncate_texts`):

   | Family | Query budget | Per-document budget |
   |---|---|---|
   | jina-reranker-v3 | 512 tokens | 2,048 tokens |
   | jina-reranker-v3.5 | 1,024 tokens | 8,192 tokens |

   A document longer than the budget is **cut** to fit (`truncation="longest_first"`).
   The engine does **not** split one long document into several scored
   passages and merge the scores. A 10k-token document on v3 is scored on its
   first 2k tokens only — the tail is invisible to ranking.

   If that matters for your data, use **client-side chunking**: split each
   document into passages ≤ the budget, rerank *passages*, and take the best
   passage score per document. The API has no chunk-aware mode; it's a
   pre-processing decision.

2. **Long candidate lists → blocked (this is the real "chunking").** In
   listwise mode (`rerank_listwise: true`) the engine packs documents into a
   prompt block and **flushes** it — one forward pass per block — when either
   the block reaches `rerank_passages_per_block` (default 16) **or** adding
   the next document would exceed the token capacity
   (`model_max_length − 2 × query_length`, which reserves room for the query
   plus readout tokens). Per-document scores are then best-kept across blocks.

   So: 1,000 documents × ~100 tokens, default block → ~63 forward passes and
   everything is scored; 1 document of 10k tokens → truncated to the family
   cap with no block fan-out to help.

## Q2. What are "blocks"? What are "passages"? How do they differ from "documents"?

- **Documents** = what you send in the request: `POST /v1/rerank { "documents": [...] }`.
  One string per candidate you want scored against the query.
- **Passages** = the Jina family's own vocabulary for the same thing. The
  paper and the reference `rerank()` call the candidate texts packed into a
  prompt "passages". In this engine they are **1:1 aliases** — there is no
  separate passage concept. Wherever you see "passage" (config, CLI, code),
  it means "document". The parameter name `passages_per_block` exists because
  the listwise mode mirrors the reference implementation, which is written in
  passage vocabulary.
- **Blocks** = a batching unit inside *listwise* scoring. The point of
  listwise is that the query attends to several documents **in one forward
  pass** — the set of documents in a single prompt is one **block**. When the
  list is longer than one prompt can hold (16 docs, or token capacity), the
  block flushes and another block begins. Each document is scored once, in
  the block it landed in; its final score is its cosine against the
  request-global aggregated query embedding (see Q4).

  Pairwise mode (`rerank_listwise: false`, the default) = one forward pass
  per document, no blocks.

## Q3. Where do the truncation and knob values come from — the model, or TabbyAPI/infinity?

Three different owners:

| Value | Owner | Nature |
|---|---|---|
| Context window (`model_max_length`) | the model (tokenizer config) | hard limit |
| Truncation caps (512/2048 vs 1024/8192) | the model — read at load from its own remote `modeling.py` via `_resolve_truncation_caps` | hard limit of the architecture |
| Block capacity = `max_length − 2×query_length` | the model's reference implementation (Last-But-Not-Late design) | reference behavior we mirror |
| 125 passages/block (reference) | the model's reference code (hard-coded there) | reference default |
| 16 passages/block (ours) | the Jina **paper** calibration (range 10–64, sweet spot 16) — chosen by us, overridable | serving knob |
| `rerank_listwise` on/off | TabbyAPI/infinity (default off = pairwise) | serving knob |
| `rerank_passages_per_block` | TabbyAPI/infinity (config / env / CLI) | serving knob |
| `embeddings_dtype`, `embeddings_device`, `top_n`, `raw_scores` | TabbyAPI/infinity | serving knobs |

Division of labor: **the model decides** how much context exists and how
prompts are structured (remote format, block flushing, cosine score
semantics) — we hold those faithful to the reference on purpose. **Our knobs
decide how you use that budget**: which mode, block size aspiration,
precision, device, and how many results to return.

Note: `passages_per_block` is an **upper bound, not a guarantee** — the
flush also triggers on token capacity, so long documents automatically shrink
blocks below your setting. The model's context is the final arbiter.

## Q4. What results should I expect with `rerank_passages_per_block: 16` and 1,000 documents of ≤100 tokens each?

From `_rerank_list_unlocked`:

- **No truncation** (docs ≪ 8,192 budget) and **no capacity flush**
  (16 × ~100 tokens ≈ 1,600 ≪ 32,768 context), so blocks are uniform:
  **62 blocks of exactly 16 + 1 block of 8 = 63 forward passes**.
- Every document is embedded exactly once, in its block, where it sees the
  other 15 documents and the query.
- The **query embedding is a request-global weighted average** over all
  blocks: each block contributes its query embedding weighted by that
  block's max normalized score (`((1+scores)/2).max()`). Final scores are
  cosine(query_avg, doc_embedding), sorted descending.
- API response: `relevance_score` in **[0, 1]** by default
  (normalized `(cos+1)/2`); with `raw_scores: true` you get the native
  cosine in **[-1, 1]**. `index` is the 0-based position in your
  `documents` array. `usage.prompt_tokens` ≈ 1,000 × doc_len + 63 ×
  query_len (~101k tokens processed across all prompts — the cost of the
  request).
- All 1,000 documents are *scored*; `top_n` only trims the *output* (see Q6).
- Duplicate documents are scored independently (no dedup).
- Caveat: because the query embedding is blended across blocks, absolute
  scores from a 1,000-doc request are not directly comparable to one from a
  3-doc request. Compare ranks within a request, and keep the mode
  consistent across your pipeline.

## Q5. Is 16 a magical, hard limit? What happens if I don't set `rerank_passages_per_block` at all?

**Not magical and not hard.** 16 is the paper's calibration sweet spot
(their sweep: block sizes 10–64 all good, 16 best for ranking quality); the
reference implementation itself hard-codes 125. The only *hard* constraints
are `passages_per_block ≥ 1` and the model's token capacity, which shrinks
blocks as needed.

**Omitting the value changes nothing** — the default chain is 16 all the way
down, so behavior is identical to writing `16` explicitly:

```
config.yml missing              → backends/infinity/model.py unwrap(..., 16)
EngineArgs.rerank_passages_per_block default = 16  (args.py field default)
validation: value < 1           → logged warning + reset to 16
rerank_list signature default   → passages_per_block = 16
```

Setting it to `0` or `-5` is caught in `args.py.__post_init__`, warns, and
resets to 16 — it can't break a boot.

Unlike `top_n` (a request field), `passages_per_block` is **read once at
model load** — it's a server-level knob (`embeddings.rerank_passages_per_block`
in config.yml, `INFINITY_RERANK_PASSAGES_PER_BLOCK`, or
`--rerank-passages-per-block` on the v2 CLI), not per-request.

Trade-offs if you tune it: halving to 8 gives more per-request averaging
granularity but doubles passes for very long lists (1,000 docs → 125 passes
instead of 63); bumping to 64 cuts passes (16) but dilutes each block's
cross-document context and long docs trip the capacity flush anyway.

## Q6. What is `top_n`?

The response-trimming knob on the `/v1/rerank` request:

```json
{ "query": "...", "documents": [...1000 docs...], "top_n": 10 }
```

→ returns the **10 highest-scoring** `(index, relevance_score)` pairs,
sorted descending. Everything else is scored but not sent back.

- **Per-request field** (`endpoints/OAI/types/rerank.py`:
  `top_n: Optional[int] = Field(None, ge=1, ...)`), unlike
  `passages_per_block`.
- **Omit it (or `null`) → ALL documents are returned**, fully sorted
  ("Returns all when omitted."). The engine's `128` helper default only
  applies if a caller passes nothing down to it; the API layer defaults to
  returning everything.
- **It trims output, not computation** — the model scores the entire
  candidate list regardless (a 1,000-doc request runs the same 63 forward
  passes and ~101k tokens whether you ask for `top_n: 10` or omit it). This
  differs from retrieval setups where `top_n` skips work.
- Validation: `ge=1` (no 0/negative); asking for more than the list length
  returns the whole list.

---

## Score semantics cheat sheet (pairwise vs listwise)

- **Pairwise** (default): each document is scored in isolation — the query
  attends to that one document. Scores from pairwise and listwise runs are
  **not directly comparable** (the query embedding differs: pairwise queries
  see one doc; listwise queries see the whole block).
- **Listwise** (`rerank_listwise: true`): query attends to all documents in
  its block; matches the reference block logic, verified block-for-block
  against `model.rerank()` for both v3 and v3.5 (including per-family
  truncation caps).
- Both modes keep **ordering** consistent on small requests; absolute
  numbers drift (measured ~0.068 on a 3-doc request) without changing rank.
- Use `embeddings_dtype: float32` for golden-grade score reproducibility:
  bf16 (the CUDA `auto` default) drifts cosine scores ~1e-3 (order
  unaffected).
