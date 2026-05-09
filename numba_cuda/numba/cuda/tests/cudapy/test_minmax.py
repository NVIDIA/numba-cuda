# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest

from numba import cuda
from numba.cuda import float64
from numba.cuda.testing import skip_on_cudasim


def builtin_max(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = float64(max(A[i], B[i]))


def builtin_min(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = float64(min(A[i], B[i]))


def _run(
    kernel,
    numpy_equivalent,
    ptx_instruction,
    dtype_left,
    dtype_right,
    n=5,
):
    kernel = cuda.jit(kernel)

    c = np.zeros(n, dtype=np.float64)
    a = np.arange(n, dtype=dtype_left) + 0.5
    b = np.full(n, fill_value=2, dtype=dtype_right)

    kernel[1, c.shape](a, b, c)
    np.testing.assert_allclose(c, numpy_equivalent(a, b))

    ptx = next(p for p in kernel.inspect_asm().values())
    assert ptx_instruction in ptx


@skip_on_cudasim("Tests PTX emission")
@pytest.mark.parametrize(
    "kernel,numpy_equivalent,ptx_instruction,dtype_left,dtype_right",
    [
        pytest.param(
            builtin_max,
            np.maximum,
            "max.f64",
            np.float64,
            np.float64,
            id="max_f8f8",
        ),
        pytest.param(
            builtin_max,
            np.maximum,
            "max.f64",
            np.float32,
            np.float64,
            id="max_f4f8",
        ),
        pytest.param(
            builtin_max,
            np.maximum,
            "max.f64",
            np.float64,
            np.float32,
            id="max_f8f4",
        ),
        pytest.param(
            builtin_max,
            np.maximum,
            "max.f32",
            np.float32,
            np.float32,
            id="max_f4f4",
        ),
        pytest.param(
            builtin_min,
            np.minimum,
            "min.f64",
            np.float64,
            np.float64,
            id="min_f8f8",
        ),
        pytest.param(
            builtin_min,
            np.minimum,
            "min.f64",
            np.float32,
            np.float64,
            id="min_f4f8",
        ),
        pytest.param(
            builtin_min,
            np.minimum,
            "min.f64",
            np.float64,
            np.float32,
            id="min_f8f4",
        ),
        pytest.param(
            builtin_min,
            np.minimum,
            "min.f32",
            np.float32,
            np.float32,
            id="min_f4f4",
        ),
    ],
)
def test_minmax(
    kernel, numpy_equivalent, ptx_instruction, dtype_left, dtype_right
):
    _run(kernel, numpy_equivalent, ptx_instruction, dtype_left, dtype_right)
