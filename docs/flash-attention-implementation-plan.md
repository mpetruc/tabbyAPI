# Flash Attention Implementation Plan (jina-reranker family)

Branch: `feat/jina-reranker-flash-attn` (off `feat/jina-reranker-v3`)

## Problem being solved (root cause, not symptom)

jina-reranker-v3/v3.5 are Qwen3-4B-derived cross-encoders. With `rerank_listwise: true`
the vendored `jina_v3.py` packs ~16 documents into a **single listwise prompt(s)**, padded;
a single rerank request with 32k-char docs produced a forward where transformers' sdpa path
materializes dense 4D attention masks of size `O(B·L²)` (bool `[B,1,L,L]` ×2 for Qwen3's
full+sliding layers, then an in-kernel fp32 conversion of the same size). Observed on the
4090/24GiB prod server: `Currently allocated 14.15 GiB`, `Requested 29.14 GiB` (the fp32 mask
conversion), `Device limit 23.54 GiB` → `torch.OutOfMemoryError`.

Flash attention (`attn_implementation="flash_attention_2"`) computes attention without
materializing `[B,1,L,L]` masks (padding handled in-kernel) → memory drops to `O(B·L)`,
which is what makes the model's 131k-token context actually usable.

## Goal

Let operators select the attention implementation for torch-engine rerankers
(JinaForRanking v3/v3.5 and generic sentence-transformers cross-encoders):

```yaml
embeddings:
  attn_implementation: flash_attention_2   # "eager" | "sdpa" | "flash_attention_2" | (unset = model default)
```

with safe, *loud* fallbacks when the choice cannot be honored.

## Shared contract (all workstreams MUST follow this exactly)

### Knob name and plumbing chain

1. `EngineArgs.attn_implementation: Optional[str] = None` (frozen dataclass,
   `infinity/libs/infinity_emb/infinity_emb/args.py`) — values `None | "eager" | "sdpa" | "flash_attention_2"`.
   Invalid value → `logger.warning` + reset to `None` (mirror the existing
   `rerank_passages_per_block` validation style in `__post_init__`).
2. Env var `INFINITY_ATTN_IMPLEMENTATION` via `infinity_emb/env.py` `MANAGER`:
   `_optional_infinity_var("attn_implementation", default="")`, empty → `None`.
   Do NOT pass the ENGINE field default as `None` through MANAGER in a way that breaks
   the existing `MANAGER.xxx[0]` access pattern — mirror how `revision` (default `""`)
   is handled and normalized in `__post_init__`.
3. tabbyAPI config: `common/config_models.py` EmbeddingConfig
   `embeddings_attn_implementation: Optional[Literal["eager","sdpa","flash_attention_2"]] = None`
   (description: what it does, flash-attn requirement, fp32 caveat — see below),
   mirrored in `endpoints/core/types/model.py::EmbeddingModelLoadRequest`
   (`attn_implementation` request field reading `config.embeddings.embeddings_attn_implementation`),
   forwarded through `backends/infinity/model.py::InfinityContainer.load` as
   `attn_implementation=kwargs.get("embeddings_attn_implementation")` → `EngineArgs(...)`.
   Keep the `unwrap()` style used for the other embeddings_* kwargs.

### Resolution & fallback semantics (single source of truth)

New module `infinity/libs/infinity_emb/infinity_emb/transformer/attention.py`:

```python
ATTN_IMPLEMENTATIONS = ("eager", "sdpa", "flash_attention_2")

def resolve_attn_implementation(requested, loading_dtype) -> Optional[str]:
    """Return the attn_implementation to pass to the model, or None (model default).

    - None/"eager"/"sdpa": passthrough.
    - "flash_attention_2" requires the `flash_attn` package (CHECK_FLASH_ATTN) AND
      a non-float32 loading dtype (FA2 kernels are fp16/bf16 only):
        * flash_attn missing      → logger.warning("falling back to sdpa ...") → "sdpa"
        * loading_dtype float32   → logger.warning("FA2 needs fp16/bf16; jina cosine scores
                                    drift ~1e-3 in bf16 (ordering unaffected);
                                    falling back to sdpa") → "sdpa"
    loading_dtype may be None (auto) → treated as acceptable (auto resolves to bf16/fp16 on GPU).
    """
```

`_optional_imports.py`: add `CHECK_FLASH_ATTN = OptionalImports("flash_attn", "torch")`.

Post-load verification helper (same module):

```python
def verify_attn_implementation(model, requested) -> None:
    """WARN if the loaded model's effective _attn_implementation != requested
    (remote-code JinaForRanking may pin its own); read
    model.config._attn_implementation (fall back to model.model.config / model.transformer.config)."""
```

### Where the resolved value is applied

- `infinity_emb/transformer/crossencoder/jina_v3.py::__init__`: add `attn_implementation`
  result into the existing `model_kwargs` dict passed to `AutoModel.from_pretrained(...)`
  (ALWAYS via `resolve_attn_implementation`, keyed by `engine_args.attn_implementation`
  and `ls.loading_dtype`). After load, call `verify_attn_implementation`.
- `infinity_emb/transformer/crossencoder/torch.py::CrossEncoderPatched.__init__`:
  same, into `automodel_args` passed to `sentence_transformers.CrossEncoder(...)`.
  Conflict rule: if `bettertransformer` is enabled, bettertransformer wins (it forces
  `attn_implementation="eager"`) — `logger.warning` if the user requested something else
  and the two interact (mirror the existing `attn_implementation="eager"` block).
- Do NOT touch: optimum/ct2/neuron/vision paths, embedders (document as out of scope).

### Runtime dependency

`flash-attn` is NOT added to pyproject extras (build complexity); it is an operator-side
install. Docs must note: `pip install flash-attn` (wheels exist for CUDA 12.x + py3.13 for
sm86/sm89 via recent flash-attn 2.7.x/2.8.x releases; sm89 (4090) requires a 2.x wheel —
flash-attn 3.x is Hopper-only). The fork already logs at boot whether the vendored
infinity_emb is live; the fallback warnings make a missing install self-explanatory.

## Workstreams (twin subagents, file-disjoint)

| WS | Owner | Files (only these) | Deliverable |
|----|-------|--------------------|-------------|
| 1 | twin-1 | `infinity/libs/infinity_emb/infinity_emb/args.py`, `env.py`, `_optional_imports.py`, `transformer/attention.py` (new), `transformer/crossencoder/jina_v3.py`, `transformer/crossencoder/torch.py` | contract items 1–2 + resolution/application |
| 2 | twin-2 | `common/config_models.py`, `endpoints/core/types/model.py`, `backends/infinity/model.py`, `config_sample.yml` | contract item 3 (tabbyAPI plumbing) |
| 3 | twin-3 | `infinity/libs/infinity_emb/tests/unit_test/test_attn_implementation.py` (new) + `tests/script_flash_smoke.py` (new, under vendored tests/) | unit tests (no GPU) + GPU smoke script (never run heavy: I run it after integration) |
| 4 | twin-4 | `README.md` (1 section), `docs/jina-reranker-v3-faq.md` (FAQ entry), `docs/flash-attention-usage.md` (new short guide) | operator docs: knob, install, fp32 caveat, expected memory O(L) |

## Constraints for every twin

- Work in `/home/dev/tabbyapi` on branch `feat/jina-reranker-flash-attn` (already checked out).
- NO `git` commands, NO `pip install`, NO running of the GPU smoke script, NO edits to files
  outside your WS column. Do not touch `.pi/`, `.github/`, `common/` (except WS2), etc.
- Only import infinity_emb symbol names that already exist; keep the fork's `logger` usage.
- After editing run `/home/dev/tabbyapi/.venv/bin/python -m py_compile <your files>` and
  report the output. Also run ruff if available (`/home/dev/tabbyapi/.venv/bin/ruff check <files>`).
- Keep every change minimal and consistent with the fork's existing style (SPDX header,
  docstrings, `logger.info/warning` phrasing already used in the same files).
- DO NOT change behavior when `attn_implementation` is unset/`None` — default path must be
  byte-identical to today (no k/v reorder, no dtype changes, no bettertransformer changes).

## Verification (architect, after all twins report)

1. `git diff` review workstream-by-workstream; conflicts resolved by me.
2. `py_compile` + ruff on all touched files.
3. Unit tests (no GPU): fallback matrix, invalid value, env var, EngineArgs passthrough.
4. GPU smoke on the dev 3080: sdpa (bf16) vs eager parity; flash_attention_2 requested
   without flash-attn installed → warning + sdpa; native-flash long single prompt (B=1,
   maskless, L≈64k bf16) → O(L) memory demonstration; jina_v3-shaped tiny Qwen3 forward.
5. If `pip install flash-attn` succeeds on the dev box: real FA2 padded-batch memory test.

## Acceptance

- Setting `embeddings.attn_implementation: flash_attention_2` (with fp16/bf16 dtype + the
  package installed) runs the prod listwise workload without the dense `[B,1,L,L]` mask
  allocation; 131k-context prompts are memory-O(L).
- Unset knob → behavior identical to `feat/jina-reranker-v3`.
- Any unsatisfiable request fails loud at load time (warning + visible fallback reason).
