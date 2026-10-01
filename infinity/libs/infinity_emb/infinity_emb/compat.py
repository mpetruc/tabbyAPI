# Copyright (c) 2023-now michaelfeil
"""
Compatibility shims for the pinned dependency matrix.

Wave 0 finding: `transformers>=4.57` stopped *exporting* `CodeCarbonCallback`
(and the codecarbon/neptune-style training integrations) through its lazy
`transformers.integrations` namespace, but `sentence-transformers` (up to and
including 3.4.1, the latest available in our index) hard-imports it in
`model_card.py`:

    from transformers.integrations import CodeCarbonCallback

With transformers 4.57.6 installed, ANY import path that touches
`sentence_transformers` (including `CrossEncoder`, used by the torch embedder
and crossencoder engines) raised ModuleNotFoundError at import time.

The removed class still lives in the real module
`transformers.integrations.integration_utils` (it is simply no longer listed
in the `_import_structure` of the lazy wrapper). We therefore re-expose it
there, and additionally plant the name directly on the lazy instance in
`sys.modules` so both resolution paths succeed.

Both packages are bound by our matrix (transformers 4.57.x is required for the
qwen3-based Jina reranker family; sentence-transformers <4.0 is a TabbyAPI
pin), so the shim is the maintained compatibility layer. The stub is only
used by sentence-transformers' model-card code for `isinstance` checks and as
a field default; no carbon tracking is performed.
"""

from typing import Any

_installed = False


def _codecarbon_callback_stub() -> Any:
    class _CodeCarbonCallbackStub:
        """Placeholder for transformers' removed CodeCarbonCallback.

        Matches the surface sentence-transformers touches: isinstance checks
        and a default field value. No carbon tracking is performed.
        """

    return _CodeCarbonCallbackStub


def install_transformers_st_compat() -> bool:
    """Re-expose `CodeCarbonCallback` for sentence-transformers.

    Tries, in order: the real `integration_utils` module, then a direct
    `__dict__` write on the lazy `transformers.integrations` instance in
    `sys.modules`. Never raises; returns True if the name was (re)installed.
    """
    global _installed
    if _installed:
        return False

    stub = _codecarbon_callback_stub()
    installed = False

    try:
        import transformers.integrations.integration_utils as integration_utils

        if not hasattr(integration_utils, "CodeCarbonCallback"):
            setattr(integration_utils, "CodeCarbonCallback", stub)
        installed = True
    except Exception:
        pass  # falls through to the sys.modules route below

    try:
        integration_module = __import__(
            "transformers.integrations", fromlist=["CodeCarbonCallback"]
        )
        if "CodeCarbonCallback" not in vars(integration_module):
            vars(integration_module)["CodeCarbonCallback"] = stub
        installed = True
    except Exception:
        pass

    _installed = installed
    return installed
