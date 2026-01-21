# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def abs_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = abs(inp[i])


@cuda.jit
def min_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = min(a[i], b[i])


@cuda.jit
def max_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = max(a[i], b[i])


@cuda.jit
def bool_kernel(inp, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = bool(inp[i])


@skip_on_cudasim("Builtins semantics differ under cudasim")
class TestCudaBuiltins(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()


    def test_abs_int(self):
        src = np.array([-1, 2, -3, 4], dtype=np.int32)
        dst = np.zeros_like(src)

        d_src = cuda.to_device(src)
        d_dst = cuda.to_device(dst)

        self._launch_1d(abs_kernel, (d_src, d_dst), src.size)

        np.testing.assert_array_equal(d_dst.copy_to_host(), np.abs(src))


    def test_abs_float(self):
        src = np.array([-1.5, 2.0, -3.25], dtype=np.float32)
        dst = np.zeros_like(src)

        d_src = cuda.to_device(src)
        d_dst = cuda.to_device(dst)

        self._launch_1d(abs_kernel, (d_src, d_dst), src.size)

        np.testing.assert_array_equal(d_dst.copy_to_host(), np.abs(src))


    def test_min(self):
        a = np.array([1, 5, 3, 7], dtype=np.int32)
        b = np.array([2, 4, 6, 0], dtype=np.int32)
        out = np.zeros_like(a)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.to_device(out)

        self._launch_1d(min_kernel, (da, db, dout), a.size)

        np.testing.assert_array_equal(dout.copy_to_host(), np.minimum(a, b))


    def test_max(self):
        a = np.array([1, 5, 3, 7], dtype=np.int32)
        b = np.array([2, 4, 6, 0], dtype=np.int32)
        out = np.zeros_like(a)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.to_device(out)

        self._launch_1d(max_kernel, (da, db, dout), a.size)

        np.testing.assert_array_equal(dout.copy_to_host(), np.maximum(a, b))


    def test_bool(self):
        src = np.array([0, 1, -1, 3], dtype=np.int32)
        dst = np.zeros(src.size, dtype=np.bool_)

        d_src = cuda.to_device(src)
        d_dst = cuda.to_device(dst)

        self._launch_1d(bool_kernel, (d_src, d_dst), src.size)

        np.testing.assert_array_equal(d_dst.copy_to_host(), src.astype(bool))
