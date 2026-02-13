# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


@skip_on_cudasim("CuPy interoperability requires a real GPU")
class TestCupyBfloat16Interop(CUDATestCase):
    def test_cupy_bfloat16_kernel_interop(self):
        if not cuda.is_bfloat16_supported():
            self.skipTest("bfloat16 requires compute capability 8.0+")

        cp = pytest.importorskip("cupy")
        ml_dtypes = pytest.importorskip("ml_dtypes")

        arr_h = np.asarray([1, 2, 3, 4, 5], dtype=ml_dtypes.bfloat16)
        arr_d = cp.asarray(arr_h)
        arr_2_d = cp.arange(5)
        arr_out_d = cp.empty(arr_d.size, dtype=ml_dtypes.bfloat16)

        @cuda.jit
        def my_ker(arr_in, arr_in2, arr_out):
            tid = cuda.grid(1)
            if tid < arr_in.size:
                arr_out[tid] = (
                    2 * arr_in[tid] + 4 * arr_in2[tid] + ml_dtypes.bfloat16(3.0)
                )

        block = 128
        grid = (arr_d.size + block - 1) // block
        my_ker[grid, block](arr_d, arr_2_d, arr_out_d)

        self.assertTrue((arr_out_d == 2 * arr_d + 4 * arr_2_d + 3.0).all())


if __name__ == "__main__":
    unittest.main()
