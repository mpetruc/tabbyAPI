# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""Fork-local compatibility shims.

The vendored engines must bridge transformers' rename of the
``from_pretrained(..., torch_dtype=...)`` kwarg to ``dtype`` (upstream
infinity_emb 0.0.77 still passes the legacy name):

- transformers >= 4.56.0 accepts ``dtype``; passing ``torch_dtype`` still
  works but logs "`torch_dtype` is deprecated! Use `dtype` instead!".
- transformers < 4.56.0 only accepts ``torch_dtype``; passing ``dtype``
  there is silently ignored because the kwarg did not exist yet.

``from_pretrained_dtype_kwarg()`` returns whichever kwarg name the installed
transformers understands, so engine loaders stay warning-free on new
transformers while still honoring the requested loading dtype on old ones.
"""

from __future__ import annotations


def from_pretrained_dtype_kwarg() -> str:
    """Return ``"dtype"`` (transformers >= 4.56) or ``"torch_dtype"`` (older)."""
    try:
        import transformers
        from packaging.version import Version

        return "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    except (ImportError, ValueError):
        # transformers not installed or version not parseable: fall back to the
        # legacy kwarg (the engine call sites only run when torch + transformers
        # are available and would hit the modern branch anyway).
        return "torch_dtype"
