# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import cuda
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
        cases = [
            np.float64([1.0, 0.0, float("inf"), float("nan")]),
            np.float64([1.0, -0.0, float("inf"), float("nan")]),
            np.float64([1.0, 1.5, float("inf"), float("nan")]),
            np.float64([[1.0, 1.5], [float("inf"), float("nan")]]),
            np.float64([[1.0, 1.5], [1.5, 1.0]]),
        ]

        case_0 = cases[0]
        case_1 = cases[1]
        case_2 = cases[2]
        case_3 = cases[3]
        case_4 = cases[4]

        @cuda.jit
        def kernel(out):
            gid = cuda.grid(1)
            if gid == 0:
                ans = np.all(case_0)
            if gid == 1:
                ans = np.all(case_1)
            if gid == 2:
                ans = np.all(case_2)
            if gid == 3:
                ans = np.all(case_3)
            if gid == 4:
                ans = np.all(case_4)
            out[gid] = ans

        expected = np.array([np.all(a) for a in cases], dtype=np.bool_)
        out = cuda.to_device(np.zeros(len(cases), dtype=np.bool_))
        kernel[1, len(cases)](out)
        got = out.copy_to_host()

        self.assertPreciseEqual(expected, got)

    def test_any_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.any(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.bool_))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.any(arr), out.copy_to_host()[0])

        arr = np.float64([0.0, -0.0, 0.0, 0.0])
        check(arr)
        arr[2] = float("nan")
        check(arr)
        arr[2] = float("inf")
        check(arr)
        arr[2] = 1.5
        check(arr)
        arr = arr.reshape((2, 2))
        check(arr)
        check(arr[::-1])

    def test_sum_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.sum(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.sum(arr), out.copy_to_host()[0])

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_mean_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.mean(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.mean(arr), out.copy_to_host()[0])

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_var_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.var(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(
                np.var(arr), out.copy_to_host()[0], prec="double"
            )

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_std_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.std(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.std(arr), out.copy_to_host()[0])

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_min_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.min(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.min(arr), out.copy_to_host()[0])

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_max_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.max(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.max(arr), out.copy_to_host()[0])

        arrays = [
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
        ]
        for arr in arrays:
            check(arr)

    def test_nanmin_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanmin(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nanmin(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nanmax_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanmax(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nanmax(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nanmean_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanmean(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nanmean(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64(
                [np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0]
            ),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nansum_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nansum(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nansum(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nanprod_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanprod(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nanprod(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nanstd_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanstd(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.nanstd(arr), out.copy_to_host()[0])

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)

    def test_nanvar_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.nanvar(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)
            self.assertPreciseEqual(
                np.nanvar(arr), out.copy_to_host()[0], prec="double"
            )

        arrays = [
            np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
            np.float64([-0.0, -1.5]),
            np.float64([-1.5, 2.5, np.nan]),
            np.float64([-1.5, 2.5, float("inf")]),
            np.float64([-1.5, 2.5, -float("inf")]),
            np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
            np.float64([np.nan, -1.5, 2.5, np.nan, 3.0]),
            np.float64([5.0, np.nan, -1.5, np.nan]),
            np.float64([np.nan, np.nan]),
        ]
        for arr in arrays:
            check(arr)
