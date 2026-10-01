# Supporting Jina-style reranker families in infinity-emb

Research notes for a session focused on the Infinity library (`infinity/` clone)
and how models like `jinaai/jina-reranker-v3.5` could be served through the
rerank path used by TabbyAPI.

---

## 0. Context: how the rerank path is reached

- TabbyAPI serves OpenAI-compatible routes backed by the **installed** `infinity-emb`
  package (pip `.[extras]`, never the clone — the clone is source reference only).
- A single `AsyncEmbeddingEngine` container loads one model; its `capabilities`
  set decides whether `/v1/embeddings` (`embed`) or `/v1/rerank` (`rerank`) work.
- TabbyAPI's `InfinityContainer` (`backends/infinity/model.py`) constructs
  `EngineArgs(model_name_or_path=<dir>, engine="torch", device=<cpu default>,
  bettertransformer=False, model_warmup=False)` — no other knobs threaded through.
- TabbyAPI has **no classify endpoint**, so models detected as classifiers
  (≥2 `id2label` labels) are unusable there.
- `/v1/rerank` request: `{query, documents, return_documents, raw_scores, top_n}`
  (Cohere protocol, mirrors Infinity's own `/rerank` route).

## 1. How Infinity decides a model is a reranker today

`inference/select_model.py::get_engine_type_from_config()` reads the model's
`config.json`:

```python
if any("SequenceClassification" in arch for arch in config.get("architectures", [])):
    id2label = config.get("id2label", {"0": "dummy"})
    if len(id2label) < 2:
        return RerankEngine            # capabilities = {"rerank"}
    else:
        return PredictEngine           # capabilities = {"classify"}
if config.get("vision_config"):
    return ImageEmbedEngine
if config.get("audio_config") and "clap" in config.get("model_type", "").lower():
    return AudioEmbedEngine
else:
    return EmbedderEngine              # capabilities = {"embed"}
```

Capability sets live on the base classes (`transformer/abstract.py`):
`BaseEmbedder {"embed"}`, `BaseCrossEncoder {"rerank"}`, classifier `{"classify"}`.

The rerank implementation (`transformer/crossencoder/torch.py`,
`CrossEncoderPatched`) is a thin wrapper around **sentence-transformers
CrossEncoder**: `encode_pre` tokenizes `(query, document)` pairs, `encode_core`
runs the forward pass and takes `logits`, `encode_post` flattens.
`BatchHandler.rerank` applies `sigmoid` (unless `raw_scores`), sorts desc, `top_n`.
A model without the `rerank` capability → `ModelNotDeployedError` → TabbyAPI maps
to HTTP 400.

Infinity's own rule of thumb (README): *"Reranking models supported by infinity
are bert-style classification models with one category."*

## 2. Verified config data for the Jina family (checked on HF, 2026-02)

| Model | `architectures` | `id2label` | `model_type` | remote code | `modules.json` (ST export) |
|---|---|---|---|---|---|
| `jinaai/jina-reranker-v3.5` | `["JinaForRanking"]` | — | `qwen3` | yes (`auto_map`: AutoModel → `modeling.JinaForRanking`; `hidden_size` 1024, 28 layers ≈ 3.7B, bf16) | no |
| `jinaai/jina-reranker-v2-base-multilingual` | `["XLMRobertaForSequenceClassification"]` | `{"0": "LABEL_0"}` (1) | *(missing/None in config)* | yes (`configuration_xlm_roberta.py`, `modeling_xlm_roberta.py`, `block.py`, `mha.py`, `mlp.py`; `auto_map` incl. `AutoModelForSequenceClassification`) | no |


All ship `onnx/*.onnx` including `int8`/`q4`/`uint8` quantized variants.

## 3. Analysis per model

**`jina-reranker-v3.5` — does not fit the current pipeline, would need real work.**
- Detection: `JinaForRanking` has no `SequenceClassification` substring → falls
  through to the **embedder** branch → `SentenceTransformer` load → no
  `modules.json` → load fails.
- Even with detection fixed, it is a **generative/decoder reranker** (qwen3
  core): uses a chat-template prompt (`query: …` / `document: …` turns) and
  scores by logit/probability extraction over generated positions — not a
  single `logits` head. The pair-tokenization + `logits[0]` + sigmoid pipeline
  does not apply. Needs either a template-aware adapter or a new capability.
- `JinaForRanking` is registered for `AutoModel` only; the cross-encoder path
  instantiates via `AutoModelForSequenceClassification`, so a custom loader /
  class mapping is required regardless of `trust_remote_code`.

**`jina-reranker-v2-base-multilingual` — closest to working today.**
- Detection **passes**: `XLMRobertaForSequenceClassification` + 1 label → RerankEngine.
- The architecture class name is standard, so a default
  `AutoModelForSequenceClassification` load may work (remote code ignored);
  however it is *intended* to run via the shipped `modeling_xlm_roberta.py`
  (flash-xlm-roberta custom attention) — behavior/quality with the stock class
  is unverified. Needs an empirical test.
- Note: `config.json` has no top-level `model_type` (printed `None`) — the
  config resolution path through `AutoConfig` should be exercised.

## 4. Gaps in Infinity to support these families

1. **Detection** (`select_model.py`): only the `SequenceClassification` substring
   + 1-label rule exists. Custom arch names (`JinaForRanking`, `JinaBertModel`)
   and modern decoder rerankers are invisible. Options to explore:
   allowlist of arch names, capability mapping file, or
   `pipeline_tag: text-classification` + single-label heuristic.
2. **Loader**: cross-encoder path is hard-wired to sentence-transformers
   `CrossEncoder` → `AutoModelForSequenceClassification`. Custom classes
   (`JinaForRanking`) and remote-code models need an adapter / dynamic class
   resolution path. Sentence-transformers' `CrossEncoder` accepts
   `trust_remote_code` (threaded through `EngineArgs`), but that only helps if
   the class name resolves in the auto-class registry.
3. **Input templating**: the pipeline tokenizes raw `(query, doc)` pairs.
   Template-driven rerankers (jina v3.x, qwen3-reranker) need chat/app templating
   before tokenization — no hook exists.
4. **Scoring semantics**: pipeline assumes one logit per pair → sigmoid.
   Generative rerankers score differently (logit extraction, tie-breaking
   handling), so `BatchHandler.rerank` output semantics need a variant or
   `raw_scores` extension.
5. **Testing**: add Jina models to `test_torch_reranker.py`
   (parametrized) and update README/docs model lists to match reality.

## 5. Existing plumbing that helps

- `EngineArgs.trust_remote_code` exists and **defaults to True**
  (`args.py:58` ← `env.py` `MANAGER.trust_remote_code` default `["true"]`);
  passed through in both `CrossEncoderPatched` and `SentenceTransformerPatched`.
- Engine backends beyond torch: `optimum` (ONNX) and `ct2` — the Jina repos
  ship `onnx/*.onnx` (fp16/int8/q4) which the optimum path could consume.
- `get_engine_type_from_config` is a single, well-isolated decision point.
- Rerank e2e tests exist (`test_torch_reranker.py`, conftest sets the model) —
  a good template for parametrizing new families.
- The `AuthCapabilities`/route layer, `ModelNotDeployedError` → 400 mapping
  already work once the engine exposes `rerank`.

## 6. Notes for the new session

- Source of truth for detection: `infinity/libs/infinity_emb/infinity_emb/inference/select_model.py`.
- Iterate in the clone, then install for real testing:
  `pip install -e ./infinity/libs/infinity_emb` (or match the pinned
  `sentence-transformers < 4.0`, `huggingface_hub < 1.0` constraints from
  TabbyAPI's `pyproject.toml`).
- Reproduce current failures first: `create_server(engine_args_list=[EngineArgs(
  model_name_or_path="jinaai/jina-reranker-v3.5", ...)])` → `POST /rerank`,
  and same for v2-base-multilingual to see whether the stock-class fallback works.
- Smallest TabbyAPI-side surface change afterwards: none needed for detection
  (automatic at load), only threading any new `EngineArgs` knobs if the loader
  strategy requires them.
- Configs re-checked at:
  - <https://huggingface.co/jinaai/jina-reranker-v3.5/raw/main/config.json>
  - <https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/raw/main/config.json>
  - <https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/raw/main/config.json>
