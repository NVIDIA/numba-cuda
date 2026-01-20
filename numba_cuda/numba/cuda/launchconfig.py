# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""Launch configuration access for CUDA compilation contexts.

The current launch configuration is populated only during CUDA compilation
triggered by kernel launches. It is thread-local and cleared immediately after
compilation completes.
"""

import contextlib
from functools import wraps

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


@contextlib.contextmanager
def capture_compile_config(dispatcher):
    """Capture the launch config seen during compilation for a dispatcher.

    The returned dict has a single key, ``"config"``, which is populated when a
    compilation is triggered by a kernel launch. If the kernel is already
    compiled, the dict value may remain ``None``.
    """
    if dispatcher is None:
        raise TypeError("dispatcher is required")

    record = {"config": None}
    original = dispatcher._compile_for_args

    @wraps(original)
    def wrapped(*args, **kws):
        record["config"] = current_launch_config()
        return original(*args, **kws)

    dispatcher._compile_for_args = wrapped
    try:
        yield record
    finally:
        dispatcher._compile_for_args = original


__all__ = [
    "current_launch_config",
    "ensure_current_launch_config",
    "capture_compile_config",
]
