# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba import jit, njit
from numba.tests.support import TestCase, MemoryLeakMixin
from numba import cuda


def array_all(arr):
    return arr.all()


def array_all_global(arr):
    return np.all(arr)


def array_any(arr):
    return arr.any()


def array_any_global(arr):
    return np.any(arr)


def array_cumprod(arr):
    return arr.cumprod()


def array_cumprod_global(arr):
    return np.cumprod(arr)


def array_nancumprod(arr):
    return np.nancumprod(arr)


def array_cumsum(arr):
    return arr.cumsum()


def array_cumsum_global(arr):
    return np.cumsum(arr)


def array_nancumsum(arr):
    return np.nancumsum(arr)


def array_sum(arr):
    return arr.sum()


def array_sum_global(arr):
    return np.sum(arr)


def array_prod(arr):
    return arr.prod()


def array_prod_global(arr):
    return np.prod(arr)


def array_mean(arr):
    return arr.mean()


def array_mean_global(arr):
    return np.mean(arr)


def array_var(arr):
    return arr.var()


def array_var_global(arr):
    return np.var(arr)


def array_std(arr):
    return arr.std()


def array_std_global(arr):
    return np.std(arr)


def array_min(arr):
    return arr.min()


def array_min_global(arr):
    return np.min(arr)


def array_amin(arr):
    return np.amin(arr)


def array_max(arr):
    return arr.max()


def array_max_global(arr):
    return np.max(arr)


def array_amax(arr):
    return np.amax(arr)


def array_argmin(arr):
    return arr.argmin()


def array_argmin_global(arr):
    return np.argmin(arr)


def array_argmax(arr):
    return arr.argmax()


def array_argmax_global(arr):
    return np.argmax(arr)


def array_median_global(arr):
    return np.median(arr)


def array_nanmin(arr):
    return np.nanmin(arr)


def array_nanmax(arr):
    return np.nanmax(arr)


def array_nanmean(arr):
    return np.nanmean(arr)


def array_nansum(arr):
    return np.nansum(arr)


def array_nanprod(arr):
    return np.nanprod(arr)


def array_nanstd(arr):
    return np.nanstd(arr)


def array_nanvar(arr):
    return np.nanvar(arr)


def array_nanmedian_global(arr):
    return np.nanmedian(arr)


def array_percentile_global(arr, q):
    return np.percentile(arr, q)


def array_nanpercentile_global(arr, q):
    return np.nanpercentile(arr, q)


def array_ptp_global(a):
    return np.ptp(a)


def array_ptp(a):
    return a.ptp()


def array_quantile_global(arr, q):
    return np.quantile(arr, q)


def array_nanquantile_global(arr, q):
    return np.nanquantile(arr, q)


def base_test_arrays(dtype):
    if dtype == np.bool_:

        def factory(n):
            assert n % 2 == 0
            return np.bool_([0, 1] * (n // 2))
    else:

        def factory(n):
            return np.arange(n, dtype=dtype) + 1

    a1 = factory(10)
    a2 = factory(10).reshape(2, 5)
    # The prod() of this array fits in a 32-bit int
    a3 = (factory(12))[::-1].reshape((2, 3, 2), order="A")
    assert not (a3.flags.c_contiguous or a3.flags.f_contiguous)

    return [a1, a2, a3]


def full_test_arrays(dtype):
    array_list = base_test_arrays(dtype)

    # Add floats with some mantissa
    if dtype == np.float32:
        array_list += [a / 10 for a in array_list]

    # add imaginary part
    if dtype == np.complex64:
        acc = []
        for a in array_list:
            tmp = a / 10 + 1j * a / 11
            tmp[::2] = np.conj(tmp[::2])
            acc.append(tmp)
        array_list.extend(acc)

    for a in array_list:
        assert a.dtype == np.dtype(dtype)
    return array_list


def run_comparative(compare_func, test_array):
    cfunc = njit(compare_func)
    numpy_result = compare_func(test_array)
    numba_result = cfunc(test_array)

    return numpy_result, numba_result


class TestArrayReductions(MemoryLeakMixin, TestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    def setUp(self):
        super(TestArrayReductions, self).setUp()
        np.random.seed(42)

    def check_reduction_basic(self, pyfunc, **kwargs):
        # Basic reduction checks on 1-d float64 arrays
        cfunc = jit(nopython=True)(pyfunc)

        def check(arr):
            self.assertPreciseEqual(pyfunc(arr), cfunc(arr), **kwargs)

        arr = np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5])
        check(arr)
        arr = np.float64([-0.0, -1.5])
        check(arr)
        arr = np.float64([-1.5, 2.5, "inf"])
        check(arr)
        arr = np.float64([-1.5, 2.5, "-inf"])
        check(arr)
        arr = np.float64([-1.5, 2.5, "inf", "-inf"])
        check(arr)
        arr = np.float64(["nan", -1.5, 2.5, "nan", 3.0])
        check(arr)
        arr = np.float64(["nan", -1.5, 2.5, "nan", "inf", "-inf", 3.0])
        check(arr)
        arr = np.float64([5.0, "nan", -1.5, "nan"])
        check(arr)
        # Only NaNs
        arr = np.float64(["nan", "nan"])
        check(arr)

    def test_all_basic(self):
        @cuda.jit
        def kernel(out):
            gid = cuda.grid(1)
            if gid < 1:
                ary = np.float64([1.0, 0.0, float("inf"), float("nan")])
                out[0] = np.all(ary)

        out = cuda.to_device(np.zeros(1, dtype=np.bool_))
        kernel[1, 1](out)

        arr = np.float64([1.0, 0.0, float("inf"), float("nan")])
        self.assertPreciseEqual(np.all(arr), out.copy_to_host()[0])

        # arr = np.float64([1.0, 0.0, float('inf'), float('nan')])
        # check(arr)
        # arr[1] = -0.0
        # check(arr)
        # arr[1] = 1.5
        # check(arr)
        # arr = arr.reshape((2, 2))
        # check(arr)
        # check(arr[::-1])
