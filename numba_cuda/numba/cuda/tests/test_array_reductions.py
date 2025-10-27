# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import cuda
from numba.cuda import literal_unroll
from numba.cuda import config


class TestArrayReductions(MemoryLeakMixin, TestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    def setUp(self):
        super(TestArrayReductions, self).setUp()
        np.random.seed(42)
        self.old_nrt_setting = config.CUDA_ENABLE_NRT
        config.CUDA_ENABLE_NRT = True

    def tearDown(self):
        config.CUDA_ENABLE_NRT = self.old_nrt_setting
        super(TestArrayReductions, self).tearDown()

    def test_all_basic(self):
        cases = (
            np.float64([1.0, 0.0, float("inf"), float("nan")]),
            np.float64([1.0, -0.0, float("inf"), float("nan")]),
            np.float64([1.0, 1.5, float("inf"), float("nan")]),
            np.float64([[1.0, 1.5], [float("inf"), float("nan")]]),
            np.float64([[1.0, 1.5], [1.5, 1.0]]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for case in literal_unroll(cases):
                out[i] = np.all(case)
                i += 1

        expected = np.array([np.all(a) for a in cases], dtype=np.bool_)
        out = cuda.to_device(np.zeros(len(cases), dtype=np.bool_))
        kernel[1, 1](out)
        got = out.copy_to_host()
        self.assertPreciseEqual(expected, got)

    def test_any_basic(self):
        cases = (
            np.float64([0.0, -0.0, 0.0, 0.0]),
            np.float64([0.0, -0.0, np.nan, 0.0]),
            np.float64([0.0, -0.0, float("inf"), 0.0]),
            np.float64([0.0, -0.0, 1.5, 0.0]),
            np.float64([[0.0, -0.0], [1.5, 0.0]]),
            np.float64([[0.0, -0.0], [1.5, 0.0]])[::-1],
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(cases):
                out[i] = np.any(arr)
                i += 1

        expected = np.array([np.any(a) for a in cases], dtype=np.bool_)
        out = cuda.to_device(np.zeros(len(cases), dtype=np.bool_))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_sum_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.sum(arr)
                i += 1

        expected = np.array([np.sum(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_mean_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.mean(arr)
                i += 1

        expected = np.array([np.mean(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_var_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.var(arr)
                i += 1

        expected = np.array([np.var(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host(), prec="double")

    def test_std_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.std(arr)
                i += 1

        expected = np.array([np.std(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_min_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.min(arr)
                i += 1

        expected = np.array([np.min(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_max_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.max(arr)
                i += 1

        expected = np.array([np.max(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_nanmin_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.nanmin(arr)
                i += 1

        expected = np.array([np.nanmin(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_nanmax_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.nanmax(arr)
                i += 1

        expected = np.array([np.nanmax(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_nanmean_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.nanmean(arr)
                i += 1

        expected = np.array([np.nanmean(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_nansum_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.nansum(arr)
                i += 1

        expected = np.array([np.nansum(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())

    def test_nanprod_basic(self):
        arrays = (
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        )

        @cuda.jit
        def kernel(out):
            i = 0
            for arr in literal_unroll(arrays):
                out[i] = np.nanprod(arr)
                i += 1

        expected = np.array([np.nanprod(a) for a in arrays], dtype=np.float64)
        out = cuda.to_device(np.zeros(len(arrays), dtype=np.float64))
        kernel[1, 1](out)
        self.assertPreciseEqual(expected, out.copy_to_host())
