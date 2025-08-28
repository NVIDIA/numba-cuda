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
