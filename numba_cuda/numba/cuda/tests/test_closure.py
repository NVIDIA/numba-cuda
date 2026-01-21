# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim

# -----------------------------
# Kernel factories with closures
# -----------------------------

def make_add_kernel(const):
    @cuda.jit
    def kernel(inp, out):
        i = cuda.grid(1)
        if i < inp.size:
            out[i] = inp[i] + const
    return kernel

def make_mul_kernel(factor):
    @cuda.jit
    def kernel(inp, out):
        i = cuda.grid(1)
        if i < inp.size:
            out[i] = inp[i] * factor
    return kernel

def make_bool_closure_kernel(flag):
    @cuda.jit
    def kernel(out):
        i = cuda.grid(1)
        if i < out.size:
            if flag:
                out[i] = 1
            else:
                out[i] = 0
    return kernel

def make_nested_constant_kernel():
    x = 3
    y = 4
    @cuda.jit
    def kernel(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = x + y
    return kernel

@skip_on_cudasim("Closure capture semantics differ under cudasim")
class TestCudaClosure(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()

    def test_closure_add_constant(self):
        add5 = make_add_kernel(5)
        inp = np.arange(10, dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(add5, (d_inp, d_out), inp.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            inp + 5,
        )

    def test_closure_multiply_constant(self):
        mul3 = make_mul_kernel(3)
        inp = np.array([1, 2, 3, 4], dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(mul3, (d_inp, d_out), inp.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            inp * 3,
        )

    def test_bool_closure_true(self):
        kernel = make_bool_closure_kernel(True)
        out = np.zeros(6, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.ones_like(out),
        )

    def test_bool_closure_false(self):
        out = np.ones(6, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.zeros_like(out),
        )

    def test_multiple_constant_capture(self):
        kernel = make_nested_constant_kernel()
        out = np.zeros(8, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 7),
        )
