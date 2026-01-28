# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import MemoryLeakMixin
from numba.cuda.testing import NRTEnablingCUDATestCase
from numba import cuda

from itertools import combinations_with_replacement
from numba.cuda.testing import skip_on_cudasim, skip_on_nvjitlink_13_1_sm_120
from numba.cuda.misc.special import literal_unroll
from numba.cuda import config
import unittest


def array_median_global(arr):
    return np.median(arr)


@skip_on_cudasim("doesn't work in the simulator")
class TestArrayReductions(MemoryLeakMixin, NRTEnablingCUDATestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    def setUp(self):
        super(TestArrayReductions, self).setUp()
        np.random.seed(42)
        self.old_nrt_setting = config.CUDA_ENABLE_NRT
        self.old_perf_warnings_setting = config.DISABLE_PERFORMANCE_WARNINGS
        config.CUDA_ENABLE_NRT = True
        config.DISABLE_PERFORMANCE_WARNINGS = 1

    def tearDown(self):
        config.CUDA_ENABLE_NRT = self.old_nrt_setting
        config.DISABLE_PERFORMANCE_WARNINGS = self.old_perf_warnings_setting
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

    @skip_on_nvjitlink_13_1_sm_120(
        "sum fails at link time on sm_120 + CUDA 13.1"
    )
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

    @skip_on_nvjitlink_13_1_sm_120(
        "mean fails at link time on sm_120 + CUDA 13.1"
    )
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

    @skip_on_nvjitlink_13_1_sm_120(
        "nanmean fails at link time on sm_120 + CUDA 13.1"
    )
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

    @skip_on_nvjitlink_13_1_sm_120(
        "nansum fails at link time on sm_120 + CUDA 13.1"
    )
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

    @skip_on_nvjitlink_13_1_sm_120(
        "nanprod fails at link time on sm_120 + CUDA 13.1"
    )
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

    def check_percentile_and_quantile(self, pyfunc, q_upper_bound):
        def check_array_q(a, q, abs_tol=1e-12):
            @cuda.jit
            def kernel(out):
                result = pyfunc(a, q)
                for i in range(len(out)):
                    out[i] = result[i]

            out = cuda.to_device(np.zeros(len(q), dtype=np.float64))
            kernel[1, 1](out)

            expected = pyfunc(a, q)
            got = out.copy_to_host()

            finite = np.isfinite(expected)
            if np.all(finite):
                self.assertPreciseEqual(got, expected, abs_tol=abs_tol)
            else:
                self.assertPreciseEqual(
                    got[finite], expected[finite], abs_tol=abs_tol
                )

        def check_scalar_q(a, q, abs_tol=1e-12):
            @cuda.jit
            def kernel(out):
                out[0] = pyfunc(a, q)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            kernel[1, 1](out)

            expected = pyfunc(a, q)
            got = out.copy_to_host()[0]

            if np.isfinite(expected):
                self.assertPreciseEqual(got, expected, abs_tol=abs_tol)

        a = self.random.randn(27).reshape(3, 3, 3)
        q = np.linspace(0, q_upper_bound, 14)[::-1].copy()

        check_array_q(a, q)
        check_scalar_q(a, 0)
        check_scalar_q(a, q_upper_bound / 2)
        check_scalar_q(a, q_upper_bound)

        not_finite = [np.nan, -np.inf, np.inf]
        a.flat[:10] = self.random.choice(not_finite, 10)
        self.random.shuffle(a)
        self.random.shuffle(q)
        check_array_q(a, q)

        a = a.flatten().tolist()
        q = q.flatten().tolist()

        # TODO - list types
        # check_array_q(a, q)
        # check(tuple(a), tuple(q))

        a = self.random.choice([1, 2, 3, 4], 10)
        q = np.linspace(0, q_upper_bound, 5)
        check_array_q(a, q)

    def test_percentile_basic(self):
        pyfunc = np.percentile
        self.check_percentile_and_quantile(pyfunc, q_upper_bound=100)
        self.check_percentile_and_quantile_edge_cases(pyfunc, q_upper_bound=100)

    @unittest.expectedFailure
    def test_percentile_exceptions(self):
        pyfunc = np.percentile
        self.check_percentile_and_quantile_exceptions(pyfunc)

    def check_percentile_and_quantile_edge_cases(
        self, pyfunc, q_upper_bound=100
    ):
        # intended to be a faitful reproduction of the upstream numba test
        # packing all the test cases into a single kernel for perf
        def _array_combinations(elements):
            for i in range(1, 10):
                for comb in combinations_with_replacement(elements, i):
                    yield np.array(comb)

        q = (0, 0.1 * q_upper_bound, 0.2 * q_upper_bound, q_upper_bound)
        element_pool = (1, -1, np.nan, np.inf, -np.inf)
        test_cases = list(_array_combinations(element_pool))

        max_len = max(len(a) for a in test_cases)
        n_cases = len(test_cases)

        # create a block containing all the test cases
        # will independently record and pass the lengths
        a_batch = np.full((n_cases, max_len), np.nan, dtype=np.float64)
        lengths = np.zeros(n_cases, dtype=np.int32)
        for i, a in enumerate(test_cases):
            a_batch[i, : len(a)] = a
            lengths[i] = len(a)

        @cuda.jit
        def kernel(a_batch, lengths, q_arr, out):
            gid = cuda.grid(1)
            if gid < a_batch.shape[0]:
                length = lengths[gid]
                a_valid = a_batch[gid, :length]
                result = np.percentile(a_valid, q_arr)
                for j in range(len(result)):
                    out[gid, j] = result[j]

        q_arr = np.array(q, dtype=np.float64)
        out = cuda.to_device(np.zeros((n_cases, len(q)), dtype=np.float64))

        kernel.forall(len(test_cases))(
            cuda.to_device(a_batch),
            cuda.to_device(lengths),
            cuda.to_device(q_arr),
            out,
        )

        got = out.copy_to_host()
        for i, a in enumerate(test_cases):
            expected = np.percentile(a, q)
            finite = np.isfinite(expected)

            if np.all(finite):
                self.assertPreciseEqual(got[i], expected, abs_tol=1e-14)
            else:
                self.assertPreciseEqual(
                    got[i][finite], expected[finite], abs_tol=1e-14
                )

    def check_percentile_and_quantile_exceptions(self, pyfunc):
        def check_scalar_q_err(a, q, abs_tol=1e-12):
            @cuda.jit
            def kernel(out):
                out[0] = np.percentile(a, q)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            with self.assertRaises(ValueError) as raises:
                kernel[1, 1](out)
            self.assertEqual(
                "Percentiles must be in the range [0, 100]",
                str(raises.exception),
            )

        # Exceptions leak references
        self.disable_leak_check()
        a = np.arange(5)
        check_scalar_q_err(a, -5)  # q less than 0
        check_scalar_q_err(a, 105)
        check_scalar_q_err(a, np.nan)

        # complex typing failure
        @cuda.jit
        def kernel(out):
            np.percentile(a, q)

        a = np.arange(5) * 1j
        q = 0.1

        out = cuda.to_device(np.zeros(1, dtype=np.float64))
        with self.assertTypingError():
            kernel[1, 1](out)

    @unittest.expectedFailure
    def check_quantile_exceptions(self, pyfunc):
        def check_scalar_q_err(a, q, abs_tol=1e-12):
            @cuda.jit
            def kernel(out):
                out[0] = np.percentile(a, q)

            out = cuda.to_device(np.zeros(1, dtype=np.float64))
            with self.assertRaises(ValueError) as raises:
                kernel[1, 1](out)
            self.assertEqual(
                "Quantiles must be in the range [0, 1]",
                str(raises.exception),
            )

        # Exceptions leak references
        self.disable_leak_check()
        a = np.arange(5)
        check_scalar_q_err(a, -0.5)  # q less than 0
        check_scalar_q_err(a, 1.05)
        check_scalar_q_err(a, np.nan)

        # complex typing failure
        @cuda.jit
        def kernel(out):
            np.quantile(a, q)

        a = np.arange(5) * 1j
        q = 0.1

        out = cuda.to_device(np.zeros(1, dtype=np.float64))
        with self.assertTypingError():
            kernel[1, 1](out)

    def test_quantile_basic(self):
        pyfunc = np.quantile
        self.check_percentile_and_quantile(pyfunc, q_upper_bound=1)
        self.check_percentile_and_quantile_edge_cases(pyfunc, q_upper_bound=1)

    def test_nanpercentile_basic(self):
        pyfunc = np.nanpercentile
        self.check_percentile_and_quantile(pyfunc, q_upper_bound=100)
        self.check_percentile_and_quantile_edge_cases(pyfunc, q_upper_bound=100)
        self.check_percentile_and_quantile_exceptions(pyfunc)

    def test_nanquantile_basic(self):
        pyfunc = np.nanquantile
        self.check_percentile_and_quantile(pyfunc, q_upper_bound=1)
        self.check_percentile_and_quantile_edge_cases(pyfunc, q_upper_bound=1)
        self.check_quantile_exceptions(pyfunc)
