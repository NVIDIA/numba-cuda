# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba import cuda
from numba.cuda import float32, int32, uint32
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def int_to_float_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = float32(inp[i])


@cuda.jit
def float64_to_float32_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = float32(inp[i])


@cuda.jit
def int_to_bool_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = bool(inp[i])


@cuda.jit
def float_to_bool_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = bool(inp[i])


@cuda.jit
def uint_to_int_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = int32(inp[i])


@cuda.jit
def implicit_promotion_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = inp[i] + 0.5



class TestNumericCasting(CUDATestCase):

    def _launch_1d(self, kernel, d_inp, d_out) -> None:
        threadsperblock = 64
        blockspergrid = (d_inp.size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](d_inp, d_out)

    @skip_on_cudasim("Casting semantics are device-specific")
    def test_int32_to_float32(self):
        # include values near float32 precision boundary (2**24)
        inp = np.array(
            [0, 1, 2**23, 2**24 - 1, 2**24, 2**24 + 1],
            dtype=np.int32,
        )
        inp = np.repeat(inp, 32)
        out = np.zeros(inp.size, dtype=np.float32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(int_to_float_kernel, d_inp, d_out)

        np.testing.assert_array_equal(d_out.copy_to_host(), inp.astype(np.float32))

    @skip_on_cudasim("Casting semantics are device-specific")
    def test_float64_to_float32(self):
        inp = np.linspace(-1e6, 1e6, 128, dtype=np.float64)
        out = np.zeros(inp.size, dtype=np.float32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(float64_to_float32_kernel, d_inp, d_out)

        np.testing.assert_allclose(
            d_out.copy_to_host(),
            inp.astype(np.float32),
            rtol=1e-7,
            atol=1e-7,
        )

    @skip_on_cudasim("Boolean casting requires device execution")
    def test_int_to_bool(self):
        inp = np.arange(-64, 64, dtype=np.int32)
        out = np.zeros(inp.size, dtype=np.bool_)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(int_to_bool_kernel, d_inp, d_out)

        np.testing.assert_array_equal(d_out.copy_to_host(), inp.astype(np.bool_))

    @skip_on_cudasim("Boolean casting requires device execution")
    def test_float_to_bool_with_nan_inf(self):
        inp = np.array(
            [-1.0, -0.0, 0.0, 0.5, np.inf, -np.inf, np.nan],
            dtype=np.float32,
        )
        inp = np.repeat(inp, 32)
        out = np.zeros(inp.size, dtype=np.bool_)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(float_to_bool_kernel, d_inp, d_out)

        np.testing.assert_array_equal(d_out.copy_to_host(), inp.astype(np.bool_))

    @skip_on_cudasim("Unsigned to signed casting semantics are device-specific")
    def test_uint32_to_int32(self):
        inp = np.array(
            [0, 1, 2, 123, 2**31 - 1, 0xFFFFFFFF],
            dtype=np.uint32,
        )
        inp = np.repeat(inp, 32)
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(uint_to_int_kernel, d_inp, d_out)

        np.testing.assert_array_equal(d_out.copy_to_host(), inp.astype(np.int32))

    @skip_on_cudasim("Implicit promotion requires device execution")
    def test_implicit_promotion_int_to_float(self):
        inp = np.arange(128, dtype=np.int32)
        out = np.zeros(inp.size, dtype=np.float32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(implicit_promotion_kernel, d_inp, d_out)

        expected = inp.astype(np.float32) + 0.5
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)
