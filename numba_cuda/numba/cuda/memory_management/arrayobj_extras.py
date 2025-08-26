# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import types
from numba.core.extending import overload_classmethod
from numba.np.arrayobj import intrin_alloc


@overload_classmethod(types.Array, "_allocate", target="CUDA")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only default target (cpu) classmethod on the array
    type.
    """

    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl
