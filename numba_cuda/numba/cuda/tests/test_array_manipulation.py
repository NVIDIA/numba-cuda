# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Array manipulation tests for CUDA device arrays.
# Covers: indexing, slicing, reshape for multiple dtypes and edge cases.
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def index_kernel(inp, out):
    # Copies each element from inp to out using grid-stride loop
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = inp[i]


@cuda.jit
def slice_kernel(inp, out, start):
    # Copies a slice from inp[start:start+len(out)] to out
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[start + i]


@cuda.jit
def reshape_kernel(inp, out, width):
    # Reshapes 1D inp to 2D out (row-major)
    i = cuda.grid(1)
    if i < out.size:
        row = i // width
        col = i % width
        out[row, col] = inp[i]


@skip_on_cudasim("Array slicing semantics differ under cudasim")
class TestCudaArrayManipulation(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()


    def test_basic_indexing(self):
        # Test for multiple dtypes and larger arrays
        for dtype in (np.int32, np.float32, np.complex64, np.uint8):
            src = np.arange(100, dtype=dtype)
            dst = np.zeros_like(src)

            d_src = cuda.to_device(src)
            d_dst = cuda.to_device(dst)

            self._launch_1d(index_kernel, (d_src, d_dst), src.size)

            np.testing.assert_array_equal(
                d_dst.copy_to_host(),
                src
            )


    def test_basic_slicing(self):
        # Test for multiple dtypes, larger arrays, and edge cases
        for dtype in (np.int32, np.float32, np.complex64, np.uint8):
            src = np.arange(50, dtype=dtype)
            # Normal slice
            start, length = 5, 20
            expected = src[start:start + length]
            out = np.zeros(length, dtype=dtype)
            d_src = cuda.to_device(src)
            d_out = cuda.to_device(out)
            self._launch_1d(slice_kernel, (d_src, d_out, start), length)
            np.testing.assert_array_equal(d_out.copy_to_host(), expected)

            # Zero-length slice
            start, length = 10, 0
            expected = src[start:start + length]
            out = np.zeros(length, dtype=dtype)
            d_src = cuda.to_device(src)
            d_out = cuda.to_device(out)
            self._launch_1d(slice_kernel, (d_src, d_out, start), length)
            np.testing.assert_array_equal(d_out.copy_to_host(), expected)


    def test_simple_reshape(self):
        # Test for multiple dtypes and larger shapes
        for dtype in (np.int32, np.float32, np.complex64, np.uint8):
            src = np.arange(60, dtype=dtype)
            reshaped = src.reshape(10, 6)
            out = np.zeros_like(reshaped)
            d_src = cuda.to_device(src)
            d_out = cuda.to_device(out)
            self._launch_1d(
                reshape_kernel,
                (d_src, d_out, reshaped.shape[1]),
                src.size,
            )
            np.testing.assert_array_equal(
                d_out.copy_to_host(),
                reshaped
            )
