import json

import pytest

from infinity_emb.args import EngineArgs
from infinity_emb.inference.select_model import (
    get_engine_type_from_config,
    select_model,
)
from infinity_emb.primitives import Device, InferenceEngine
from infinity_emb.transformer.utils import PredictEngine, RerankEngine


@pytest.mark.parametrize("engine", [e for e in InferenceEngine if e != InferenceEngine.neuron])
def test_engine(engine):
    select_model(
        EngineArgs(
            engine=engine,
            model_name_or_path=(pytest.DEFAULT_BERT_MODEL),
            batch_size=4,
            device=Device.cpu,
            model_warmup=False,
        )
    )


def _write_config(tmp_path, config: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_get_engine_type_jina_reranker_v3(tmp_path):
    """the JinaForRanking (LBNL family) architecture must map to the
    dedicated jina_v3 engine, not the generic crossencoder path."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking"],
            "id2label": {"0": "dummy"},
        },
    )
    assert (
        get_engine_type_from_config(EngineArgs(model_name_or_path=model_dir))
        == RerankEngine.jina_v3
    )


def test_get_engine_type_jina_takes_precedence_over_seqcls(tmp_path):
    """JinaForRanking must win even when SequenceClassification is listed
    alongside it in `architectures`."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking", "Qwen3ForCausalLM"],
            "id2label": {"0": "dummy"},
        },
    )
    assert (
        get_engine_type_from_config(EngineArgs(model_name_or_path=model_dir))
        == RerankEngine.jina_v3
    )


def test_get_engine_type_seqcls_unchanged(tmp_path):
    """the generic crossencoder route must be untouched for other models."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
            "id2label": {"0": "0", "1": "1"},
        },
    )
    assert (
        get_engine_type_from_config(EngineArgs(model_name_or_path=model_dir))
        == PredictEngine.torch
    )
