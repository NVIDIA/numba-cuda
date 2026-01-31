# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import config, float32, void
from numba.cuda.testing import unittest, CUDATestCase

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp


def generate_input(n):
    A = cp.array(np.arange(n * n).reshape(n, n), dtype=np.float32)
    B = cp.array(np.arange(n) + 0, dtype=A.dtype)
    return A, B


class TestCudaNonDet(CUDATestCase):
    def test_for_pre(self):
        """Test issue with loop not running due to bad sign-extension at the for
        loop precondition.
        """

        @cuda.jit(void(float32[:, :], float32[:, :], float32[:]))
        def diagproduct(c, a, b):
            startX, startY = cuda.grid(2)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            height = c.shape[0]
            width = c.shape[1]

            for x in range(startX, width, (gridX)):
                for y in range(startY, height, (gridY)):
                    c[y, x] = a[y, x] * b[x]

        N = 8

        dA, dB = generate_input(N)
        dF = cp.empty(dA.shape, dtype=dA.dtype)

        blockdim = (32, 8)
        griddim = (1, 1)

        diagproduct[griddim, blockdim](dF, dA, dB)

        E = np.dot(
            dA.get() if not config.ENABLE_CUDASIM else dA,
            np.diag(dB.get() if not config.ENABLE_CUDASIM else dB),
        )
        np.testing.assert_array_almost_equal(
            dF.get() if not config.ENABLE_CUDASIM else dF, E
        )


if __name__ == "__main__":
    unittest.main()
