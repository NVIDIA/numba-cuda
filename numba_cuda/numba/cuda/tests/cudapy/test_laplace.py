# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import float64, void
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.core import config
import cupy as cp

# NOTE: CUDA kernel does not return any value

if config.ENABLE_CUDASIM:
    tpb = 4
else:
    tpb = 16
SM_SIZE = tpb, tpb


class TestCudaLaplace(CUDATestCase):
    def test_laplace_small(self):
        @cuda.jit(float64(float64, float64), device=True, inline="always")
        def get_max(a, b):
            if a > b:
                return a
            else:
                return b

        @cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
        def jocabi_relax_core(A, Anew, error):
            err_sm = cuda.shared.array(SM_SIZE, dtype=float64)

            ty = cuda.threadIdx.x
            tx = cuda.threadIdx.y
            bx = cuda.blockIdx.x
            by = cuda.blockIdx.y

            n = A.shape[0]
            m = A.shape[1]

            i, j = cuda.grid(2)

            err_sm[ty, tx] = 0
            if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
                Anew[j, i] = 0.25 * (
                    A[j, i + 1] + A[j, i - 1] + A[j - 1, i] + A[j + 1, i]
                )
                err_sm[ty, tx] = Anew[j, i] - A[j, i]

            cuda.syncthreads()

            # max-reduce err_sm vertically
            t = tpb // 2
            while t > 0:
                if ty < t:
                    err_sm[ty, tx] = get_max(err_sm[ty, tx], err_sm[ty + t, tx])
                t //= 2
                cuda.syncthreads()

            # max-reduce err_sm horizontally
            t = tpb // 2
            while t > 0:
                if tx < t and ty == 0:
                    err_sm[ty, tx] = get_max(err_sm[ty, tx], err_sm[ty, tx + t])
                t //= 2
                cuda.syncthreads()

            if tx == 0 and ty == 0:
                error[by, bx] = err_sm[0, 0]

        if config.ENABLE_CUDASIM:
            NN, NM = 4, 4
            iter_max = 20
        else:
            NN, NM = 256, 256
            iter_max = 1000

        A = np.zeros((NN, NM), dtype=np.float64)
        Anew = np.zeros((NN, NM), dtype=np.float64)

        n = NN

        tol = 1.0e-6
        error = 1.0

        for j in range(n):
            A[j, 0] = 1.0
            Anew[j, 0] = 1.0

        iter = 0

        blockdim = (tpb, tpb)
        griddim = (NN // blockdim[0], NM // blockdim[1])

        error_grid = np.zeros(griddim)

        stream = cupy.cuda.stream()

        with stream:
            dA = cp.asarray(A)  # to device and don't come back
            dAnew = cp.asarray(Anew)  # to device and don't come back
        
            derror_grid = cp.asarray(error_grid)

            while error > tol and iter < iter_max:
                self.assertTrue(error_grid.dtype == np.float64)

                jocabi_relax_core[griddim, blockdim, stream](dA, dAnew, derror_grid)

                error_grid = derror_grid.get()

                # error_grid is available on host
                stream.synchronize()

                error = np.abs(error_grid).max()

                # swap dA and dAnew
                tmp = dA
                dA = dAnew
                dAnew = tmp

                iter += 1


if __name__ == "__main__":
    unittest.main()
