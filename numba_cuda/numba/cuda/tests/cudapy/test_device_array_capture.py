# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Tests for capturing device arrays (objects implementing __cuda_array_interface__)
from global scope in CUDA kernels and device functions.

This tests the capture of arrays that implement __cuda_array_interface__:
- Numba device arrays (cuda.to_device)
- ForeignArray (wrapper implementing __cuda_array_interface__)
"""

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim
import cupy as cp


def make_numba_array(host_arr):
    """Create a Numba device array from host array."""
    return cp.asarray(host_arr)


def make_foreign_array(host_arr):
    """Create a ForeignArray wrapping a Numba device array."""
    return ForeignArray(cp.asarray(host_arr))


def get_host_data(arr):
    """Copy array data back to host."""
    if isinstance(arr, ForeignArray):
        return arr._arr.get()
    return arr.get()


# Array factories to test: (name, factory)
ARRAY_FACTORIES = [
    ("numba_device", make_numba_array),
    ("foreign", make_foreign_array),
]


@skip_on_cudasim("Global device array capture not supported in simulator")
class TestDeviceArrayCapture(CUDATestCase):
    """Test capturing device arrays from global scope."""

    def test_basic_capture(self):
        """Test basic global capture with different array types."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_data = np.array(
                    [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                )
                global_array = make_array(host_data)

                @cuda.jit(device=True)
                def read_global(idx):
                    return global_array[idx]

                @cuda.jit
                def kernel(output):
                    i = cuda.grid(1)
                    if i < output.size:
                        output[i] = read_global(i)

                n = len(host_data)
                output = cp.zeros(n, dtype=np.float32)
                kernel[1, n](output)

                result = output.get()
                np.testing.assert_array_equal(result, host_data)

    def test_computation(self):
        """Test captured global arrays used in computations."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_data = np.array(
                    [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                )
                global_array = make_array(host_data)

                @cuda.jit(device=True)
                def double_global_value(idx):
                    return global_array[idx] * 2.0

                @cuda.jit
                def kernel(output):
                    i = cuda.grid(1)
                    if i < output.size:
                        output[i] = double_global_value(i)

                n = len(host_data)
                output = cp.zeros(n, dtype=np.float32)
                kernel[1, n](output)

                result = output.get()
                expected = host_data * 2.0
                np.testing.assert_array_equal(result, expected)

    def test_mutability(self):
        """Test that captured arrays can be written to (mutability)."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_data = np.zeros(5, dtype=np.float32)
                mutable_array = make_array(host_data)

                @cuda.jit
                def write_kernel():
                    i = cuda.grid(1)
                    if i < 5:
                        mutable_array[i] = float(i + 1)

                write_kernel[1, 5]()

                result = get_host_data(mutable_array)
                expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
                np.testing.assert_array_equal(result, expected)

    def test_multiple_arrays(self):
        """Test capturing multiple arrays from globals."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                host_b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
                arr_a = make_array(host_a)
                arr_b = make_array(host_b)

                @cuda.jit(device=True)
                def add_globals(idx):
                    return arr_a[idx] + arr_b[idx]

                @cuda.jit
                def kernel(output):
                    i = cuda.grid(1)
                    if i < output.size:
                        output[i] = add_globals(i)

                output = cp.zeros(3, dtype=np.float32)
                kernel[1, 3](output)

                result = output.get()
                expected = np.array([11.0, 22.0, 33.0], dtype=np.float32)
                np.testing.assert_array_equal(result, expected)

    def test_multidimensional(self):
        """Test capturing multidimensional arrays."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_2d = np.array(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
                )
                arr_2d = make_array(host_2d)

                @cuda.jit(device=True)
                def read_2d(row, col):
                    return arr_2d[row, col]

                @cuda.jit
                def kernel(output):
                    i = cuda.grid(1)
                    if i < 6:
                        row = i // 2
                        col = i % 2
                        output[i] = read_2d(row, col)

                output = cp.zeros(6, dtype=np.float32)
                kernel[1, 6](output)

                result = output.get()
                expected = host_2d.flatten()
                np.testing.assert_array_equal(result, expected)

    def test_dtypes(self):
        """Test capturing arrays with different dtypes."""
        dtypes = [
            (np.int32, [10, 20, 30, 40]),
            (np.float64, [1.5, 2.5, 3.5, 4.5]),
        ]

        for name, make_array in ARRAY_FACTORIES:
            for dtype, values in dtypes:
                with self.subTest(array_type=name, dtype=dtype):
                    host_data = np.array(values, dtype=dtype)
                    global_arr = make_array(host_data)

                    @cuda.jit(device=True)
                    def read_arr(idx):
                        return global_arr[idx]

                    @cuda.jit
                    def kernel(output):
                        i = cuda.grid(1)
                        if i < output.size:
                            output[i] = read_arr(i)

                    output = cp.zeros(len(host_data), dtype=dtype)
                    kernel[1, len(host_data)](output)
                    np.testing.assert_array_equal(
                        output.get(), host_data
                    )

    def test_direct_kernel_access(self):
        """Test direct kernel access (not via device function)."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_data = np.array([7.0, 8.0, 9.0], dtype=np.float32)
                global_direct = make_array(host_data)

                @cuda.jit
                def direct_access_kernel(output):
                    i = cuda.grid(1)
                    if i < output.size:
                        output[i] = global_direct[i] + 1.0

                output = cp.zeros(3, dtype=np.float32)
                direct_access_kernel[1, 3](output)

                result = output.get()
                expected = np.array([8.0, 9.0, 10.0], dtype=np.float32)
                np.testing.assert_array_equal(result, expected)

    def test_zero_dimensional(self):
        """Test capturing 0-D (scalar) device arrays."""
        for name, make_array in ARRAY_FACTORIES:
            with self.subTest(array_type=name):
                host_0d = np.array(42.0, dtype=np.float32)
                global_0d = make_array(host_0d)

                @cuda.jit
                def kernel_0d(output):
                    output[()] = global_0d[()] * 2.0

                output = cp.zeros((), dtype=np.float32)
                kernel_0d[1, 1](output)

                result = output.get()
                expected = 84.0
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
