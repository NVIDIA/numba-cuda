# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


from numba import cuda
import numpy as np

from numba.cuda.types import bfloat16
from numba.cuda import config
import pytest

if not config.ENABLE_CUDASIM:
    from numba.cuda._internal.cuda_fp8 import (
        fp8_e5m2,
        fp8_e4m3,
        fp8_e8m0,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_fp8_bindings_from_float(dtype):
    @cuda.jit
    def test_fp8_ctor(data):
        x = data[0]

        fp8_e5m2_x = fp8_e5m2(x)  # noqa: F841
        fp8_e4m3_x = fp8_e4m3(x)  # noqa: F841
        fp8_e8m0_x = fp8_e8m0(x)  # noqa: F841

    data = np.array([1.0], dtype=dtype)
    test_fp8_ctor[1, 1](data)


def test_fp8_bindings_from_bf16():
    @cuda.jit
    def test_fp8_bf16():
        x = bfloat16(1.0)

        fp8_e5m2_x = fp8_e5m2(x)  # noqa: F841
        fp8_e4m3_x = fp8_e4m3(x)  # noqa: F841
        fp8_e8m0_x = fp8_e8m0(x)  # noqa: F841

    test_fp8_bf16[1, 1]()
