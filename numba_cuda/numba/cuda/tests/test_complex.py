# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def complex_add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


@cuda.jit
def complex_mul_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] * b[i]


@cuda.jit
def complex_abs_kernel(inp, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = abs(inp[i])


@skip_on_cudasim("Complex semantics differ under cudasim")
class TestCudaComplex(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()


    def test_complex_add(self):
        a = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex64)
        b = np.array([2-1j, 4+3j, -5+6j], dtype=np.complex64)
        out = np.zeros_like(a)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.to_device(out)

        self._launch_1d(complex_add_kernel, (da, db, dout), a.size)

        np.testing.assert_array_equal(
            dout.copy_to_host(),
            a + b
        )


    def test_complex_multiply(self):
        a = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex64)
        b = np.array([2-1j, 4+3j, -5+6j], dtype=np.complex64)
        out = np.zeros_like(a)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.to_device(out)

        self._launch_1d(complex_mul_kernel, (da, db, dout), a.size)

        np.testing.assert_array_equal(
            dout.copy_to_host(),
            a * b
        )


    def test_complex_abs(self):
        inp = np.array([3+4j, 5-12j, -8+15j], dtype=np.complex64)
        out = np.zeros(inp.size, dtype=np.float32)

        dinp = cuda.to_device(inp)
        dout = cuda.to_device(out)

        self._launch_1d(complex_abs_kernel, (dinp, dout), inp.size)

        np.testing.assert_allclose(
            dout.copy_to_host(),
            np.abs(inp),
            rtol=1e-6
        )
