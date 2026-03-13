# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
import sys


@cuda.jit(cache=True)
def cache_kernel(x):
    x[0] = 1


def launch(blockdim):
    arr = np.zeros(1, dtype=np.int32)
    cache_kernel[1, blockdim](arr)
    return arr


def self_test():
    mod = sys.modules[__name__]
    out = mod.launch(32)
    assert out[0] == 1
    out = mod.launch(64)
    assert out[0] == 1
