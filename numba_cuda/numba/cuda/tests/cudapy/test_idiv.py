# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import config, float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp


class TestCudaIDiv(CUDATestCase):
    def test_inplace_div(self):
        @cuda.jit(void(float32[:, :], int32, int32))
        def div(grid, l_x, l_y):
            for x in range(l_x):
                for y in range(l_y):
                    grid[x, y] /= 2.0

        grid = cp.ones((2, 2), dtype=np.float32)
        div[1, 1](grid, 2, 2)
        y = grid.get()
        self.assertTrue(np.all(y == 0.5))

    def test_inplace_div_double(self):
        @cuda.jit(void(float64[:, :], int32, int32))
        def div_double(grid, l_x, l_y):
            for x in range(l_x):
                for y in range(l_y):
                    grid[x, y] /= 2.0

        grid = cp.ones((2, 2), dtype=np.float64)
        div_double[1, 1](grid, 2, 2)
        y = grid.get()
        self.assertTrue(np.all(y == 0.5))


if __name__ == "__main__":
    unittest.main()
