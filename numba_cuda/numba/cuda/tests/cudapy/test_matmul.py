# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda import float32, void
from numba.cuda.core import config

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp

# Ensure the test takes a reasonable amount of time in the simulator
if config.ENABLE_CUDASIM:
    bpg, tpb = 2, 8
else:
    bpg, tpb = 50, 32

n = bpg * tpb
SM_SIZE = (tpb, tpb)


def test_cuda_matmul():
    @cuda.jit(void(float32[:, ::1], float32[:, ::1], float32[:, ::1]))
    def cu_square_matrix_mul(A, B, C):
        sA = cuda.shared.array(shape=SM_SIZE, dtype=float32)
        sB = cuda.shared.array(shape=(tpb, tpb), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        x = tx + bx * bw
        y = ty + by * bh

        acc = float32(0)  # forces all the math to be f32
        for i in range(bpg):
            if x < n and y < n:
                sA[ty, tx] = A[y, tx + i * tpb]
                sB[ty, tx] = B[ty + i * tpb, x]

            cuda.syncthreads()

            if x < n and y < n:
                for j in range(tpb):
                    acc += sA[ty, j] * sB[j, tx]

            cuda.syncthreads()

            if x < n and y < n:
                C[y, x] = acc

    np.random.seed(42)
    A = np.array(np.random.random((n, n)), dtype=np.float32)
    B = np.array(np.random.random((n, n)), dtype=np.float32)
    C = np.empty_like(A)

    stream = cp.cuda.Stream()
    nb_stream = cuda.api.external_stream(stream.ptr)
    with stream:
        dA = cp.asarray(A)
        dB = cp.asarray(B)
        dC = cp.asarray(C)

    cu_square_matrix_mul[(bpg, bpg), (tpb, tpb), nb_stream](dA, dB, dC)
    with stream:
        C = dC.get() if not config.ENABLE_CUDASIM else dC

    # Host compute
    Cans = np.dot(A, B)

    # Check result
    np.testing.assert_allclose(C, Cans, rtol=1e-5)
