# Flash Attention for Torch Rerankers — Operator Guide

Applies to the vendored infinity-emb engine (`infinity/libs/infinity_emb`)
serving the Jina reranker family (jina-reranker-v3 / v3.5) and generic
sentence-transformers cross-encoders over `POST /v1/rerank`. A single knob,
`embeddings_attn_implementation`, selects the attention implementation used
when the reranker model loads; `flash_attention_2` removes the dense
attention-mask allocation that caused listwise OOMs on long prompts.

Companion docs: [jina-reranker-v3-faq.md](jina-reranker-v3-faq.md) (semantics
and knobs, incl. the listwise block model),
[flash-attention-implementation-plan.md](flash-attention-implementation-plan.md)
(design and acceptance criteria).

---

## What it does, and why

During a forward pass, transformers materializes an attention mask of shape
`[B, 1, L, L]` — **quadratic in the prompt length L**. For Qwen3-derived
models (v3/v3.5) there are **two** of these (one per full-attention layer set,
one for the sliding-window layers), and the sdpa path then performs an
in-kernel fp32 conversion of the same size. The model weights are constant;
the mask is what grows:

| Prompt length (block tokens) | Mask entries (L²) | fp32 mask conversion |
|---|---|---|
| 8k | 6.7e7 | ~0.25 GiB |
| 32k | 1.1e9 | ~4 GiB |
| 88k (observed failing case) | 7.8e9 | ~29 GiB → **OOM on a 24 GiB card** |
| 131k (model context cap) | 1.7e10 | ~68 GiB |

Observed in production with `rerank_listwise: true` and 32k-char documents,
on a 24 GiB card: `Currently allocated 14.15 GiB`, then
`Requested 29.14 GiB` (the fp32 mask conversion) against
`Device limit 23.54 GiB` → `torch.OutOfMemoryError`.

Flash attention 2 computes attention **without materializing the dense
`[B, 1, L, L]` mask at all** — padding is handled inside the kernel, so
memory drops to `O(B · L)`. The model's 131k-token context becomes actually
usable: a full 16-document listwise block at the context cap is memory-O(L)
instead of memory-O(L²).

This knob applies to **torch-engine rerankers only** (JinaForRanking v3/v3.5
and sentence-transformers cross-encoders). Embedders and the
optimum/ct2/neuron/vision paths are unaffected.

## When you need it (and when you don't)

The O(L²) mask only bites when L is large **within a single forward pass**.
That happens in listwise mode when blocks fill up with long documents —
16 documents × up to 8,192 tokens each (v3.5 cap) ≈ a 131k-token block.
Pairwise mode, short documents, and truncated lists keep L small and are fine
on the default path — **if your documents can be split client-side into
chunks of ≤ 2–4k tokens, that *is* the fix** and you never need flash
attention (nor its fp16/bf16 constraints). Chunking is complementary to this
knob, not a substitute requirement either way.

## Installing flash-attn

`flash-attn` is an opt-in **`flash` extra** (it builds from source — see
below — so it must not be in the default paths):

```bash
uv pip install ".[extras,cu13,flash]"
```

How it works, so the semantics aren't magic:

- flash-attn publishes **no Python 3.13 wheels**, so uv builds the 2.8.x
  sdist. Its `setup.py` imports `torch` at build time without declaring it as
  a build dependency — which is why target-env builds fail with
  `ModuleNotFoundError: No module named 'torch'`.
- The repo injects the **same cu130 torch wheels as the `cu13` extra** into
  the isolated build environment via `[tool.uv.extra-build-dependencies]`
  (kept in sync with `cu13`), so the extension compiles against the exact
  headers it runs with.
- The host still needs a **CUDA toolkit** (`nvcc`) and a C/C++ compiler, plus
  time — expect a 10–20 min first build. Tune it:
  ```bash
  export TORCH_CUDA_ARCH_LIST="8.9"   # sm89/4090 only → minutes, not an hour
  export MAX_JOBS=${MAX_JOBS:-4}      # don't let the build exhaust RAM
  uv pip install ".[extras,cu13,flash]"
  ```
- **cu12-stack users**: the build-env injection mirrors `cu13` only; on the
  cu12 stack install manually so the build sees your torch:
  `uv pip install --no-build-isolation "flash-attn==2.8.*"`.
- **sm89 (RTX 4090)**: stay on the 2.x line the extra pins — flash-attn
  3.x is Hopper-only (sm90).
- **macOS / CPU / non-NVIDIA**: flash-attn cannot build/run; the engine
  falls back to `sdpa` with a visible warning (see below).

## Config knob

`config.yml` → `embeddings:` (the same block as `embeddings_dtype` and
`rerank_listwise`):

```yaml
embeddings:
  embeddings_dtype: auto            # flash needs fp16/bf16 (see table below)
  rerank_listwise: true             # where the long prompts come from
  embeddings_attn_implementation: flash_attention_2   # eager | sdpa | flash_attention_2 | (unset)
```

Omit the key (or set it to `eager`/`sdpa`) to keep current behavior — the
default path is byte-identical to before this knob existed. It is read once
at model load, like `rerank_passages_per_block`.

## API payload field

To set it per-model at load time (overrides the config default), send it on
the `POST /v1/model/embedding/load` payload:

```json
{
  "embedding_model_name": "jina-reranker-v3.5",
  "embeddings_attn_implementation": "flash_attention_2"
}
```

The plumbing chain is exactly:

```
config.yml  embeddings.embeddings_attn_implementation
    └── /v1/model/embedding/load payload field embeddings_attn_implementation
          └── EngineArgs.attn_implementation   (env: INFINITY_ATTN_IMPLEMENTATION)
                └── resolve_attn_implementation() → model_kwargs["attn_implementation"]
```

An invalid value (anything outside `eager | sdpa | flash_attention_2`) is
logged and reset to `None` (model default) — it cannot break a boot, mirroring
the `rerank_passages_per_block` validation style.

## Dtype interplay

Flash attention kernels are **fp16/bf16 only**. The engine checks your
`embeddings_dtype` at load:

| `embeddings_dtype` | `flash_attention_2` result | Notes |
|---|---|---|
| `auto` (default) | **works** | auto resolves to bf16/fp16 on CUDA |
| `float16` | **works** | |
| `bfloat16` | **works** | jina cosine scores drift ~1e-3 vs float32; **ranking order unaffected** (same guidance as the FAQ cheat sheet) |
| `float32` | **falls back to `sdpa`** + warning | golden-grade reproducibility wins over memory; keep chunking for long docs |

## Verifying it took effect

At model load the service logs the resolved choice, then exactly one of the
following confirmations:

- **Resolved (always logged, INFO)** –
  `attention implementation for <model>: flash_attention_2` (or
  `model default` when the knob is unset) plus, after the weights load,
  `attention implementation 'flash_attention_2' active after load.` — this is
  the success signal.
- **flash-attn missing** –
  `attn_implementation="flash_attention_2" requested but the \`flash-attn\` package is not installed (pip install flash-attn); falling back to sdpa.`
- **float32 block** –
  `attn_implementation="flash_attention_2" requires fp16/bf16 kernels, but the model is loading in float32 (jina cosine scores drift ~1e-3 in bf16, ordering unaffected); falling back to sdpa.`
- **Post-load mismatch** (remote-code `JinaForRanking` can pin its own
  implementation; the engine reads `model.config._attn_implementation` and
  warns) –
  `attention implementation mismatch: requested 'flash_attention_2' but the model loaded with 'eager'. Some remote/quantized implementations pin their own attention; scores are still valid, but the memory profile of flash attention is not active.`

If you see the two warning lines with no visible exception, reranking
continues on `sdpa` — the request fails **loud at load**, not mid-request.

Advanced: the vendored engine's unit tests and a GPU smoke script live at
`infinity/libs/infinity_emb/tests/`
(`unit_test/test_attn_implementation.py`, `script_flash_smoke.py`) and
exercise the fallback matrix and native-flash long-prompt path directly.

## Memory expectations

| Implementation | Peak attention memory | Notes |
|---|---|---|
| `eager` | O(B·L²) fp32 scores + masks | worst; what you get with `bettertransformer` on the generic CrossEncoder path (it pins `attn_implementation="eager"` and wins over any request) |
| `sdpa` | O(B·L²) masks (fused kernel) | default path; fine for small-L blocks, was the OOM path at ~88k-token blocks |
| `flash_attention_2` | **O(B·L)**, maskless | objective: full 131k-token listwise blocks fit the 24 GiB card |

Two guard-rails that remain unchanged:

1. **Truncation still applies** — per-family token budgets (512/2,048 for v3,
   1,024/8,192 for v3.5) cap each *document* regardless of attention
   implementation. Flash attention is about the *block* not fitting the card,
   not about documents.
2. **Score semantics are unchanged** — still cosine, still `[0, 1]`
   normalized (or `[-1, 1]` with `raw_scores: true`); flash vs sdpa vs eager
   differ in memory profile, not in the listwise block logic.
