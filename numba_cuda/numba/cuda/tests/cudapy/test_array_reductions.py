# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import MemoryLeakMixin
from numba.cuda.testing import NRTEnablingCUDATestCase
from numba import cuda

from itertools import combinations_with_replacement
from numba.cuda.misc.special import literal_unroll
from numba.cuda import config


def array_median_global(arr):
    return np.median(arr)


class TestArrayReductions(MemoryLeakMixin, NRTEnablingCUDATestCase):
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

    def test_median_basic(self):
        def variations(a):
            # Sorted, reversed, random, many duplicates
            yield a
            a = a[::-1].copy()
            yield a
            np.random.shuffle(a)
            yield a
            a[a % 4 >= 1] = 3.5
            yield a

        self.check_median_basic(array_median_global, variations)

    def check_median_basic(self, pyfunc, array_variations):
        # cfunc = jit(nopython=True)(pyfunc)

        def check(arr):
            @cuda.jit
            def kernel(out):
                out[0] = np.median(arr)

            expected = pyfunc(arr)
            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)

            got = out.copy_to_host()[0]

            self.assertPreciseEqual(expected, got)

        # Empty array case
        check(np.array([]))

        # Odd sizes
        def check_odd(a):
            check(a)
            a = a.reshape((9, 7))
            check(a)
            check(a.T)

        for a in array_variations(np.arange(63) + 10.5):
            check_odd(a)

        # Even sizes
        def check_even(a):
            check(a)
            a = a.reshape((4, 16))
            check(a)
            check(a.T)

        for a in array_variations(np.arange(64) + 10.5):
            check_even(a)

    def check_percentile(self, pyfunc, q_upper_bound):
        def check(a, q, abs_tol=1e-12):
            @cuda.jit
            def kernel(out):
                result = np.percentile(a, q)
                for i in range(len(out)):
                    out[i] = result[i]

            out = cuda.to_device(np.zeros(len(q), dtype=np.float64))
            kernel[1, 1](out)

            expected = np.percentile(a, q)
            got = out.copy_to_host()

            finite = np.isfinite(expected)
            if np.all(finite):
                self.assertPreciseEqual(got, expected, abs_tol=abs_tol)
            else:
                self.assertPreciseEqual(
                    got[finite], expected[finite], abs_tol=abs_tol
                )

        a = self.random.randn(27).reshape(3, 3, 3)
        q = np.linspace(0, q_upper_bound, 14)[::-1]

        check(a, q)
        check(a, 0)
        check(a, q_upper_bound / 2)
        check(a, q_upper_bound)

        not_finite = [np.nan, -np.inf, np.inf]
        a.flat[:10] = self.random.choice(not_finite, 10)
        self.random.shuffle(a)
        self.random.shuffle(q)
        check(a, q)

        a = a.flatten().tolist()
        q = q.flatten().tolist()
        check(a, q)
        check(tuple(a), tuple(q))

        a = self.random.choice([1, 2, 3, 4], 10)
        q = np.linspace(0, q_upper_bound, 5)
        check(a, q)

        # tests inspired by
        # https://github.com/numpy/numpy/blob/345b2f6e/numpy/lib/tests/test_function_base.py
        x = np.arange(8) * 0.5
        np.testing.assert_equal(np.percentile(x, 0), 0.0)
        np.testing.assert_equal(np.percentile(x, q_upper_bound), 3.5)
        np.testing.assert_equal(np.percentile(x, q_upper_bound / 2), 1.75)

        x = np.arange(12).reshape(3, 4)
        q = np.array((0.25, 0.5, 1.0)) * q_upper_bound
        np.testing.assert_equal(np.percentile(x, q), [2.75, 5.5, 11.0])

        x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        q = np.array((0.25, 0.50)) * q_upper_bound
        np.testing.assert_equal(np.percentile(x, q).shape, (2,))

        q = np.array((0.25, 0.50, 0.75)) * q_upper_bound
        np.testing.assert_equal(np.percentile(x, q).shape, (3,))

        x = np.arange(12).reshape(3, 4)
        np.testing.assert_equal(np.percentile(x, q_upper_bound / 2), 5.5)
        self.assertTrue(np.isscalar(np.percentile(x, q_upper_bound / 2)))

        np.testing.assert_equal(np.percentile([1, 2, 3], 0), 1)

        a = np.array([2, 3, 4, 1])
        np.percentile(a, [q_upper_bound / 2])
        np.testing.assert_equal(a, np.array([2, 3, 4, 1]))

    def test_percentile_basic(self):
        pyfunc = np.percentile
        self.check_percentile(pyfunc, q_upper_bound=100)
        # self.check_percentile_edge_cases(pyfunc, q_upper_bound=100)
        # self.check_percentile_exceptions(pyfunc)

    def check_percentile_edge_cases(self, pyfunc, q_upper_bound=100):
        def check(a, q, abs_tol=1e-14):
            @cuda.jit
            def kernel(out):
                result = np.percentile(a, q)
                for i in range(len(out)):
                    out[i] = result[i]

            out = cuda.to_device(np.zeros(len(q), dtype=np.float64))
            kernel[1, 1](out)
            expected = np.percentile(a, q)

            got = out.copy_to_host()
            finite = np.isfinite(expected)

            if np.all(finite):
                self.assertPreciseEqual(got, expected, abs_tol=abs_tol)
            else:
                self.assertPreciseEqual(
                    got[finite], expected[finite], abs_tol=abs_tol
                )

        def convert_to_float_and_check(a, q, abs_tol=1e-14):
            expected = pyfunc(a, q).astype(np.float64)
            got = np.percentile(a, q)
            self.assertPreciseEqual(got, expected, abs_tol=abs_tol)

        def _array_combinations(elements):
            for i in range(1, 10):
                for comb in combinations_with_replacement(elements, i):
                    yield np.array(comb)

        # high number of combinations, many including non-finite values
        q = (0, 0.1 * q_upper_bound, 0.2 * q_upper_bound, q_upper_bound)
        element_pool = (1, -1, np.nan, np.inf, -np.inf)
        for a in _array_combinations(element_pool):
            check(a, q)

        # edge cases - numpy exhibits behavioural differences across
        # platforms, see: https://github.com/numpy/numpy/issues/13272
        if q_upper_bound == 1:
            _check = convert_to_float_and_check
        else:
            _check = check

        a = np.array(5)
        q = np.array(1)
        _check(a, q)

        a = 5
        q = q_upper_bound / 2
        _check(a, q)

    def check_percentile_exceptions(self, pyfunc):
        # TODO
        pass

    def check_quantile_exceptions(self, pyfunc):
        # TODO
        pass
