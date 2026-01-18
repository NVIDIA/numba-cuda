# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""Launch configuration access for CUDA compilation contexts.

The current launch configuration is populated only during CUDA compilation
triggered by kernel launches. It is thread-local and cleared immediately after
compilation completes.
"""

from numba.cuda.cext import _dispatcher


def current_launch_config():
    """Return the current launch configuration, or None if not set."""
    return _dispatcher.get_current_launch_config()


def ensure_current_launch_config():
    """Return the current launch configuration or raise if not set."""
    config = current_launch_config()
    if config is None:
        raise RuntimeError("No launch config set for this thread")
    return config


__all__ = ["current_launch_config", "ensure_current_launch_config"]
