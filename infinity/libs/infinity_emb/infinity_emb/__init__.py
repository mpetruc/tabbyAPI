# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import importlib.metadata

# install dependency-matrix compat shims before any heavy import
# (see infinity_emb/compat.py; Wave 0 matrix finding for transformers 4.57.x)
from infinity_emb.compat import install_transformers_st_compat  # noqa: E402

install_transformers_st_compat()

from infinity_emb.args import EngineArgs  # noqa: E402
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray  # noqa: E402
from infinity_emb.env import MANAGER  # noqa: E402

# reexports
from infinity_emb.infinity_server import create_server  # noqa: E402
from infinity_emb.log_handler import logger  # noqa: E402
from infinity_emb.sync_engine import SyncEngineArray  # noqa: E402

__version__: str = importlib.metadata.version("infinity_emb")

__all__ = [
    "__version__",
    "AsyncEmbeddingEngine",
    "AsyncEngineArray",
    "create_server",
    "EngineArgs",
    "logger",
    "MANAGER",
    "SyncEngineArray",
]
