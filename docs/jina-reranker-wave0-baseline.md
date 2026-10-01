# Wave 0 — Environment baseline record (Jina reranker v3 feature)

Date: 2026-10-01 · Branches: `feat/jina-reranker-v3` (TabbyAPI `9b89bdc`,
infinity `1eb4396` + Wave-0 commits) · Unique venv: `/home/dev/tabbyapi/.venv`

## 1. Resolved matrix (uv-only; authoritative)

| Package | Version | Note |
|---|---|---|
| python | 3.13.5 | `uv venv .venv --python 3.13` |
| torch | 2.14.1+cpu | cpu index `https://download.pytorch.org/whl/cpu` |
| torchvision | 0.29.1+cpu | **must** be the `+cpu` build — PyPI's identical version is CUDA-linked and crashes op registration with `+cpu` torch |
| transformers | 4.57.6 | matches the PR #674 lock target and Jina's declared `transformers_version: 4.57.3` |
| tokenizers | 0.22.2 | matches PR relock |
| sentence-transformers | 3.3.1 | pinned = PR matrix (TabbyAPI cap `<4.0`) |
| numpy | 2.5.3 | requires clone pyproject bound relax `>=1.20.0,<2` → `>=1.20.0,<3` (1.x has no cp313 wheels) |
| infinity-emb | 0.0.77 | editable install of `infinity/libs/infinity_emb[torch]` works via uv (poetry-layout pyproject parsed fine) |
| pytest | 8.4.2 | **pytest 9.x breaks collection** (`Marks cannot be applied to fixtures`) — keep `<9` |
| huggingface_hub | <1.0 | TabbyAPI pin honored |

Install sequence used (idempotent):
```
uv venv .venv --python 3.13
uv pip install "torch>=2.9" --index-url https://download.pytorch.org/whl/cpu
uv pip install "torchvision==0.29.1+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e "infinity/libs/infinity_emb[torch]" "sentence-transformers==3.3.1" \
    "huggingface_hub<1.0" "numpy>=2" "pytest>=8,<9" pytest-mock httpx asgi-lifespan anyio trio \
    "coverage[toml]" "requests==2.32.3" "types-requests==2.28.1" openai jinja2 jinja2-cli \
    fastapi==0.115.2 prometheus-fastapi-instrumentator==7.0.0 einops aiohttp timm uvicorn soundfile
uv pip check   # clean, 66 installed
```

## 2. Shipped Wave-0 code (infinity branch)

- `infinity_emb/compat.py` (new): `install_transformers_st_compat()` — re-exposes
  `CodeCarbonCallback`, which **transformers 4.57 removed from its lazy
  `transformers.integrations` export list** but which sentence-transformers
  (≤3.4.1, latest available) hard-imports in `model_card.py`. Without this
  shim, ANY `sentence_transformers` import (embedder + CrossEncoder paths)
  raises `ModuleNotFoundError` with transformers 4.57.6. Patches the real
  `integration_utils` module + plants the name on the lazy `sys.modules`
  instance (both routes tested).
- `infinity_emb/__init__.py`: calls the shim before other imports.
- `pyproject.toml`: `numpy >=1.20.0,<2` → `>=1.20.0,<3` (cp313; ONNX/optimum
  path to be re-evaluated separately, see risks).

## 3. Baseline test results (unit_test, `-m "not performance"`)

`63 passed, 15 failed, 1 skipped, 1 deselected, 16 warnings` (~15 min).

**Green (feature-relevant):** all torch crossencoder tests, `test_batch_handler`,
`test_select_model` (torch rows), engine reranker tests, embedder tests,
server/API tests, CLI help tests (with venv on PATH), audio tests.

**Failed 15 — full taxonomy (0 feature-relevant):**
- 5 × `test_models.py::test_bert[...]` — `ctranslate2` not installed (extra
  intentionally out of matrix).
- 4 × `test_optimum_*` (classifier, crossencoder ×2, embedder) — `optimum`
  engine not installed (out of matrix; torch-only decision).
- 2 × `test_select_model.py::test_engine[ctranslate2|optimum]` — ditto.
- 2 × `test_cli.py::test_cli_preload[v1|v2]` — CLI `--preload-only` subprocess
  exits 1 without a model/config; pre-existing, triage at G1.
- ignored: `tests/unit_test/transformer/vision/` — `colpali-engine` caps
  `torch<2.9`; our torch is 2.14 — vision extra deliberately out of matrix.
- collection-skipped: `tests/unit_test/test_infinity_server.py` — broken at the
  pinned commit (`infinity_emb.cli` exposes `v1`/`v2` as class command methods,
  module-level import fails) — pre-existing upstream mismatch.

## 4. Environment gotchas (documented for subagents)

- Run everything from the activated venv **with `$VIRTUAL_ENV/bin` on PATH**
  (CLI subprocess tests need the `infinity_emb` console script).
- Do NOT run `uv run`/`uv sync` at the TabbyAPI repo root: it resolves the
  TabbyAPI *project* (torch cu12 + exllamav3, multi-GB). Use
  `uv pip...` / `.venv/bin/python` directly, or `cd` into the clone.
- Do NOT install torch/torchvision from PyPI (CUDA builds; ABI mismatch).

## 5. Decisions carried into the plan

1. Matrix: transformers **4.57.6** (target of W-A requirements-uv.in) + ST
   **3.3.1** + the compat shim. 4.56.x does NOT exist in our index — the shim
   is the compatibility layer, not a version downgrade.
2. NumPy bound relax + pytest `<9` + torchvision `+cpu` pin are part of the
   maintained lock story (W-A).
3. Vision/optimum/ct2 extras stay out of the matrix for this feature; any
   future enablement must resolve the colpali `torch<2.9` cap first.
4. `test_infinity_server.py` collection breakage is pre-existing — W-A may fix
   the test import or keep it ignored; not feature work.
