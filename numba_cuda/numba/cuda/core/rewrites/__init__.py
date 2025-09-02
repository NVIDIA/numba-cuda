# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
A subpackage hosting Numba IR rewrite passes.
"""

from .registry import register_rewrite, rewrite_registry, Rewrite

# Register various built-in rewrite passes
from numba.cuda.core.rewrites import (
    static_getitem,
    static_raise,
    static_binop,
    ir_print,
)

__all__ = (
    "static_getitem",
    "static_raise",
    "static_binop",
    "ir_print",
    "register_rewrite",
    "rewrite_registry",
    "Rewrite",
)
