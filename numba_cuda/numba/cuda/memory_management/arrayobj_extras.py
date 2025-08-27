# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import types
from numba.core.extending import overload_classmethod
from numba.np.arrayobj import intrin_alloc


def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only CUDA-target classmethod on the array type."""

    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl


_initialized = False


def initialize():
    # We defer the initialization of the overload until it's required to avoid
    # side-effects on import
    global _initialized
    if _initialized:
        return
    print("Initializing")
    # Define overload for Array._allocate
    overload_classmethod(types.Array, "_allocate", target="CUDA")(
        _ol_array_allocate
    )
    _initialized = True
