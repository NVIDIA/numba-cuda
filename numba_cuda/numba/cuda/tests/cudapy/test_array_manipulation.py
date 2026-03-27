# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
CUDA array manipulation tests using kernels and Python CUDA semantics.

These tests validate array operations, indexing, and element-wise operations
in CUDA kernels. They are inspired by CPU array manipulation test suites and
adapted to verify CUDA device semantics.
"""

import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase


@cuda.jit
def fill_kernel(arr, value):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = value


@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


@cuda.jit
def set_index(a):
    a[3] = 42


@cuda.jit
def cast_int_to_float(src, dst):
    i = cuda.grid(1)
    if i < src.size:
        dst[i] = src[i]


@cuda.jit
def cast_float_to_int(src, dst):
    i = cuda.grid(1)
    if i < src.size:
        dst[i] = int(src[i])


@cuda.jit
def add_mixed(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


class TestArrayManipulation(CUDATestCase):
    """Tests for array manipulation operations in CUDA kernels."""

    def test_fill_basic(self):
        """Basic elementwise fill operation"""
        n = 128
        arr = cuda.device_array(n, dtype=np.float32)

        threads = 64
        blocks = (n + threads - 1) // threads
        fill_kernel[blocks, threads](arr, 4.25)

        result = arr.copy_to_host()
        np.testing.assert_array_equal(result, np.full(n, 4.25))

    def test_elementwise_add(self):
        """Elementwise addition in a CUDA kernel."""
        n = 256
        a = np.arange(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.device_array_like(a)

        threads = 64
        blocks = (n + threads - 1) // threads
        add_kernel[blocks, threads](da, db, dout)

        result = dout.copy_to_host()
        np.testing.assert_allclose(result, a + b)

    def test_integer_getitem_setitem(self):
        """Direct integer indexing in a kernel."""
        arr = cuda.device_array(10, dtype=np.int32)
        set_index[1, 1](arr)
        host = arr.copy_to_host()
        self.assertEqual(host[3], 42)

    def test_multidimensional_indexing(self):
        """2D indexing with a grid-based kernel."""
        shape = (8, 8)
        host = np.zeros(shape, dtype=np.int32)
        dev = cuda.to_device(host)

        @cuda.jit
        def write_2d(a):
            x, y = cuda.grid(2)
            if x < a.shape[0] and y < a.shape[1]:
                a[x, y] = x * a.shape[1] + y

        threads = (4, 4)
        blocks = ((shape[0] + threads[0] - 1) // threads[0],
                  (shape[1] + threads[1] - 1) // threads[1])

        write_2d[blocks, threads](dev)
        result = dev.copy_to_host()
        self.assertEqual(result[2, 3], 2 * shape[1] + 3)

    def test_dtype_transitions(self):
        """Type casting and mixed-dtype operations in CUDA kernels."""
        n = 10
        a = np.arange(n, dtype=np.int32)
        b = np.arange(n, dtype=np.float32)
        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout1 = cuda.device_array(n, dtype=np.float32)
        dout2 = cuda.device_array(n, dtype=np.int32)

        threads = 32
        blocks = (n + threads - 1) // threads
        cast_int_to_float[blocks, threads](da, dout1)
        cast_float_to_int[blocks, threads](db, dout2)

        np.testing.assert_allclose(dout1.copy_to_host(), a.astype(np.float32))
        np.testing.assert_array_equal(dout2.copy_to_host(), b.astype(np.int32))

        # Mixed dtype elementwise op
        dout3 = cuda.device_array(n, dtype=np.float32)
        add_mixed[blocks, threads](da, db, dout3)
        np.testing.assert_allclose(dout3.copy_to_host(), a + b)

    def test_shape_semantics_in_kernel(self):
        """Shape attributes are accessible inside a kernel."""
        shape = (5, 7)
        arr = cuda.device_array(shape, dtype=np.int32)

        @cuda.jit
        def check_shape(a, out):
            if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
                out[0] = a.shape[0]
                out[1] = a.shape[1]

        out = cuda.device_array(2, dtype=np.int32)
        check_shape[1, 1](arr, out)
        host = out.copy_to_host()
        self.assertEqual(tuple(host), shape)

    def test_grid_stride_correctness(self):
        """Grid-stride loop pattern."""
        n = 10
        arr = cuda.device_array(n, dtype=np.int32)

        @cuda.jit
        def grid_stride_kernel(a):
            for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
                a[i] = i

        # Intentionally over-provisioned: more threads than elements
        # to verify the grid-stride loop handles the wrap-around.
        threads = 32
        blocks = 2
        grid_stride_kernel[blocks, threads](arr)
        host = arr.copy_to_host()
        np.testing.assert_array_equal(host, np.arange(n))

