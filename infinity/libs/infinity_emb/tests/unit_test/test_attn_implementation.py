"""Unit tests for the attention-implementation knob (CPU-only, no model loads).

Covers the shared contract in ``docs/flash-attention-implementation-plan.md``
(workstream 3):

* :func:`~infinity_emb.transformer.attention.resolve_attn_implementation`
  fallback matrix: ``None``/``eager``/``sdpa`` pass through untouched;
  ``flash_attention_2`` degrades *loudly* to ``sdpa`` when ``flash-attn`` is
  missing or the loading dtype is float32 (FA2 kernels are fp16/bf16 only).
* ``EngineArgs.attn_implementation`` validation. Explicit values are tested
  in-process; the env-var/default behavior is probed in a subprocess so the
  ``MANAGER`` env cache is read in a fresh interpreter.
* :func:`~infinity_emb.transformer.attention.verify_attn_implementation`
  post-load verification on direct and wrapped model configs.

No GPU is required and no model is ever loaded.
"""

import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from infinity_emb.args import EngineArgs
from infinity_emb.transformer import attention
from infinity_emb.transformer.attention import (
    ATTN_IMPLEMENTATIONS,
    resolve_attn_implementation,
    verify_attn_implementation,
)


class _FakeFlashCheck:
    """Stand-in for ``OptionalImports`` exposing only ``is_available``.

    ``OptionalImports.is_available`` is a ``cached_property`` (a data
    descriptor), so it cannot be monkeypatched on the ``CHECK_FLASH_ATTN``
    instance; swapping the module-level binding in
    ``infinity_emb.transformer.attention`` is the deterministic way to
    control it.
    """

    def __init__(self, available: bool) -> None:
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available


@pytest.fixture
def fake_flash_check(monkeypatch):
    def _set(available: bool) -> None:
        monkeypatch.setattr(attention, "CHECK_FLASH_ATTN", _FakeFlashCheck(available))

    return _set


def _warning_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _no_warnings(caplog) -> bool:
    return not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_attn_implementations_tuple():
    assert ATTN_IMPLEMENTATIONS == ("eager", "sdpa", "flash_attention_2")


@pytest.mark.parametrize(
    "requested,loading_dtype",
    [
        (None, None),
        (None, torch.float32),
        (None, torch.bfloat16),
        (None, torch.float16),
        ("eager", None),
        ("eager", torch.float32),
        ("eager", torch.bfloat16),
        ("eager", torch.float16),
        ("sdpa", None),
        ("sdpa", torch.float32),
        ("sdpa", torch.bfloat16),
        ("sdpa", torch.float16),
    ],
)
def test_none_eager_sdpa_pass_through(caplog, requested, loading_dtype):
    """None/eager/sdpa pass through unchanged for any loading dtype."""
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    assert resolve_attn_implementation(requested, loading_dtype) is requested
    assert _no_warnings(caplog)


def test_flash_missing_package_falls_back_to_sdpa(caplog, fake_flash_check):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    fake_flash_check(False)
    assert resolve_attn_implementation("flash_attention_2", torch.bfloat16) == "sdpa"
    messages = _warning_messages(caplog)
    assert any("falling back to sdpa" in m for m in messages)
    assert any("flash-attn" in m for m in messages)


def test_flash_float32_loading_falls_back_to_sdpa(caplog, fake_flash_check):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    fake_flash_check(True)
    assert resolve_attn_implementation("flash_attention_2", torch.float32) == "sdpa"
    messages = _warning_messages(caplog)
    assert any("fp16/bf16" in m for m in messages)
    assert any("falling back to sdpa" in m for m in messages)


@pytest.mark.parametrize("loading_dtype", [torch.bfloat16, torch.float16, None])
def test_flash_honored_with_fp16_bf16_or_auto(caplog, fake_flash_check, loading_dtype):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    fake_flash_check(True)
    assert (
        resolve_attn_implementation("flash_attention_2", loading_dtype)
        == "flash_attention_2"
    )
    assert _no_warnings(caplog)


def test_resolve_rejects_invalid_value():
    assert resolve_attn_implementation("bogus", torch.bfloat16) is None


# --------------------------------------------------------------------------- #
# EngineArgs.attn_implementation
# --------------------------------------------------------------------------- #


def test_engine_args_explicit_value():
    args = EngineArgs(attn_implementation="flash_attention_2")
    assert args.attn_implementation == "flash_attention_2"


def test_engine_args_invalid_value_warns_and_resets(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    args = EngineArgs(attn_implementation="bogus")
    assert args.attn_implementation is None
    assert any("is invalid" in m for m in _warning_messages(caplog))


def test_engine_args_empty_value_resets_silently(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    args = EngineArgs(attn_implementation="")
    assert args.attn_implementation is None
    assert _no_warnings(caplog)


# ``attn_implementation`` defaults to ``MANAGER.attn_implementation``, which is
# an env-cached singleton read at import time. To cover the default (unset) and
# the INFINITY_ATTN_IMPLEMENTATION env var deterministically, probe fresh
# interpreters in a subprocess with a scrubbed environment (no INFINITY_* vars
# leak in from the outer run).
_SUBPROCESS_PROBE = """
import importlib
import os

import infinity_emb.args as args_mod
import infinity_emb.env as env_mod


def probe(label):
    importlib.reload(env_mod)
    importlib.reload(args_mod)
    from infinity_emb.args import EngineArgs

    value = EngineArgs(device="cpu").attn_implementation
    print(f"{label}={value!r}")


probe("clean")
os.environ["INFINITY_ATTN_IMPLEMENTATION"] = "eager"
probe("env_eager")
os.environ["INFINITY_ATTN_IMPLEMENTATION"] = ""
probe("env_empty")
os.environ["INFINITY_ATTN_IMPLEMENTATION"] = "bogus"
probe("env_bogus")
"""


def test_env_var_and_default_via_subprocess():
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFINITY_")}
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    # unset env (or empty string) -> model default (None)
    assert lines["clean"] == "None"
    assert lines["env_empty"] == "None"
    # valid env value is honored
    assert lines["env_eager"] == "'eager'"
    # invalid env value is reset to None with a warning
    assert lines["env_bogus"] == "None"
    assert "is invalid" in proc.stderr


# --------------------------------------------------------------------------- #
# verify_attn_implementation
# --------------------------------------------------------------------------- #


def test_verify_matching_impl_no_warning(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    verify_attn_implementation(model, "sdpa")
    assert _no_warnings(caplog)


def test_verify_mismatch_warns(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    verify_attn_implementation(model, "sdpa")
    messages = _warning_messages(caplog)
    assert any("mismatch" in m and "sdpa" in m and "eager" in m for m in messages)


def test_verify_wrapped_model_config(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"))
    )
    verify_attn_implementation(model, "flash_attention_2")
    assert _no_warnings(caplog)
    verify_attn_implementation(model, "sdpa")
    assert any("mismatch" in m for m in _warning_messages(caplog))


def test_verify_transformer_wrapped_config(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    model = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    )
    verify_attn_implementation(model, "eager")
    assert _no_warnings(caplog)


def test_verify_no_config_does_not_crash(caplog):
    caplog.set_level(logging.WARNING, logger="infinity_emb")
    # neither model.config nor model.model/transformer.config present
    verify_attn_implementation(SimpleNamespace(unrelated=1), "sdpa")
    assert _no_warnings(caplog)
    # requested=None short-circuits before touching the config at all
    caplog.clear()
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    verify_attn_implementation(model, None)
    assert _no_warnings(caplog)
