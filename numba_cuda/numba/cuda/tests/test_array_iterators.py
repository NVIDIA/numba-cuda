# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


import numpy as np

from numba.cuda import jit
from numba import typeof
from numba.core import types
from numba.cuda.tests.support import MemoryLeakMixin, override_config
from numba.cuda.testing import CUDATestCase
import unittest
import itertools
from numba.cuda import config

if config.ENABLE_CUDASIM:
    raise unittest.SkipTest("Array iterator tests not done in simulator")


def array_iter(arr, out):
    total = 0
    for i, v in enumerate(arr):
        total += i * v
    out[0] = total


def array_view_iter(arr, idx, out):
    total = 0
    for i, v in enumerate(arr[idx]):
        total += i * v
    out[0] = total


def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v


def array_flat_getitem(arr, ind, out):
    out[0] = arr.flat[ind]


def array_flat_setitem(arr, ind, val):
    arr.flat[ind] = val


def array_flat_sum(arr, out):
    s = 0
    for i, v in enumerate(arr.flat):
        s = s + (i + 1) * v
    out[0] = s


def array_flat_len(arr, out):
    out[0] = len(arr.flat)


def array_ndenumerate_sum(arr, out):
    s = 0
    for (i, j), v in np.ndenumerate(arr):
        s = s + (i + 1) * (j + 1) * v
    out[0] = s


def np_ndindex_empty(out):
    s = 0
    for ind in np.ndindex(()):
        s += s + len(ind) + 1
    out[0] = s


def np_ndindex(x, y, out):
    s = 0
    n = 0  # noqa: F841
    for i, j in np.ndindex(x, y):
        s = s + (i + 1) * (j + 1)
    out[0] = s


def np_ndindex_array(arr, out):
    s = 0
    n = 0  # noqa: F841
    for indices in np.ndindex(arr.shape):
        for i, j in enumerate(indices):
            s = s + (i + 1) * (j + 1)
    out[0] = s


def np_nditer1a(a, out):
    for u in np.nditer(a):
        out = u.item()  # noqa: F841


def np_nditer1b(a, out):
    i = 0
    for u in np.nditer(a):
        out[i] = u.item()
        i += 1


def np_nditer2a(a, b, out):
    for u, v in np.nditer((a, b)):
        out[0] = u.item()
        out[1] = v.item()


def np_nditer2b(a, b, out):
    i = 0
    for u, v in np.nditer((a, b)):
        out[i] = u.item()
        out[i + 1] = v.item()
        i += 2


def np_nditer2b_err(a, b, out):
    i = 0
    for u, v in np.nditer((a, b)):
        out[i] = u.item()
        out[i + 1] = v.item()
        i += 2


def np_nditer3(a, b, c, out):
    i = 0
    for u, v, w in np.nditer((a, b, c)):
        out[i] = u.item()
        out[i + 1] = v.item()
        out[i + 2] = w.item()
        i += 3


def iter_next(arr, out):
    it = iter(arr)
    it2 = iter(arr)
    out[0] = next(it)
    out[1] = next(it)
    out[2] = next(it2)


#
# Test premature free (see issue #2112).
# The following test allocates an array ``x`` inside the body.
# The compiler will put a ``del x`` right after the last use of ``x``,
# which is right after the creation of the array iterator and
# before the loop is entered.  If the iterator does not incref the array,
# the iterator will be reading garbage data of free'ed memory.
#


def array_flat_premature_free(size, out):
    x = np.arange(size)
    res = np.zeros_like(x, dtype=np.intp)
    for i, v in enumerate(x.flat):
        res[i] = v
    for i in range(len(res)):
        out[i] = res[i]


def array_ndenumerate_premature_free(size, out):
    x = np.arange(size)
    res = np.zeros_like(x, dtype=np.intp)
    for i, v in np.ndenumerate(x):
        res[i] = v
    for i in range(len(res)):
        out[i] = res[i]


class TestArrayIterators(MemoryLeakMixin, CUDATestCase):
    """
    Test array.flat,  etc.
    """

    def setUp(self):
        super(TestArrayIterators, self).setUp()

    def check_array_iter_1d(self, arr):
        out = np.zeros(1, dtype=np.int32)
        cout = np.zeros(1, dtype=np.int32)
        pyfunc = array_iter
        cfunc = jit((typeof(arr), typeof(out)))(pyfunc)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out[0], cout[0])

    def check_array_view_iter(self, arr, index):
        out = np.zeros(1)
        cout = np.zeros(1)
        pyfunc = array_view_iter
        cfunc = jit(
            (
                typeof(arr),
                typeof(index),
                typeof(out),
            )
        )(pyfunc)
        pyfunc(arr, index, out)
        cfunc[1, 1](arr, index, cout)
        self.assertPreciseEqual(out[0], cout[0])

    def check_array_flat(self, arr, arrty=None):
        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()
        if arrty is None:
            arrty = typeof(arr)

        cfunc = jit(
            (
                arrty,
                typeof(out),
            )
        )(array_flat)

        array_flat(arr, out)
        cfunc[1, 1](arr, nb_out)

        self.assertPreciseEqual(out, nb_out)

    def check_array_unary(self, arr, arrty, func):
        out = np.zeros(3)
        cout = np.zeros(3)
        cfunc = jit((arrty, typeof(out)))(func)
        func(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)

    def check_array_ndenumerate_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_ndenumerate_sum)

    def test_array_iter(self):
        # Test iterating over arrays
        arr = np.arange(6)
        self.check_array_iter_1d(arr)
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.check_array_iter_1d(np.ascontiguousarray(arr))
        arr = np.bool_([1, 0, 0, 1])
        self.check_array_iter_1d(arr)

    def test_array_view_iter(self):
        # Test iterating over a 1d view over a 2d array
        arr = np.arange(12).reshape((3, 4))
        self.check_array_view_iter(arr, 1)
        self.check_array_view_iter(arr.T, 1)
        arr = arr[::2]
        self.check_array_view_iter(np.ascontiguousarray(arr), 1)
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_view_iter(arr, 1)

    def test_array_flat_3d(self):
        arr = np.arange(24).reshape(4, 2, 3)

        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, "C")
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_flat(arr)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, "F")
        self.check_array_flat(arr)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, "A")
        self.check_array_flat(np.ascontiguousarray(arr))
        # Boolean array
        arr = np.bool_([1, 0, 0, 1] * 2).reshape((2, 2, 2))
        self.check_array_flat(np.ascontiguousarray(arr))

    def test_array_flat_empty(self):
        # Test .flat with various shapes of empty arrays, contiguous
        # and non-contiguous (see issue #846).

        # Define a local checking function, Numba's `typeof` ends up aliasing
        # 0d C and F ordered arrays, so the check needs to go via the compile
        # result entry point to bypass type checking.
        def check(arr, arrty):
            out = np.zeros(1, dtype=np.int32)
            cout = np.zeros(1, dtype=np.int32)
            cfunc = jit((arrty, typeof(out)))(array_flat_sum)
            array_flat_sum(arr, out)
            cfunc[1, 1](arr, cout)
            self.assertPreciseEqual(out[0], cout[0])

        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout="C")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="F")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="A")
        check(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout="C")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="F")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="A")
        check(arr, arrty)

    def test_array_flat_getitem(self):
        # Test indexing of array.flat object
        pyfunc = array_flat_getitem
        cfunc = jit(pyfunc)

        def check(arr, ind):
            out = np.zeros(1, dtype=np.int32)
            cout = np.zeros(1, dtype=np.int32)
            pyfunc(arr, ind, out)
            cfunc[1, 1](arr, ind, cout)
            self.assertEqual(cout[0], out[0])

        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(np.ascontiguousarray(arr), i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        # Boolean array
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(np.ascontiguousarray(arr), i)

    def test_array_flat_setitem(self):
        # Test indexing of array.flat object
        pyfunc = array_flat_setitem
        cfunc = jit(pyfunc)

        def check(arr, ind):
            # Use np.copy() to keep the layout
            expected = np.copy(arr)
            got = np.copy(arr)
            pyfunc(expected, ind, 123)
            cfunc[1, 1](got, ind, 123)
            self.assertPreciseEqual(got, expected)

        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(np.ascontiguousarray(arr), i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)
        # Boolean array
        arr = np.bool_([1, 0, 0, 1])
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)

    def test_array_flat_len(self):
        # Test len(array.flat)
        pyfunc = array_flat_len
        cfunc = jit(array_flat_len)

        def check(arr):
            out = np.zeros(1, dtype=np.int32)
            cout = np.zeros(1, dtype=np.int32)
            pyfunc(arr, out)
            cfunc[1, 1](arr, cout)
            self.assertEqual(cout[0], out[0])

        arr = np.arange(24).reshape(4, 2, 3)
        check(arr)
        arr = arr.T
        check(arr)
        arr = np.array([42]).reshape(())
        check(arr)

    def test_array_flat_premature_free(self):
        with override_config("CUDA_ENABLE_NRT", True):
            out = np.zeros(6)
            cout = np.zeros(6)
            cfunc = jit((types.intp, typeof(out)))(array_flat_premature_free)
            array_flat_premature_free(6, out)
            cfunc[1, 1](6, cout)
            self.assertTrue(cout.sum())
            self.assertPreciseEqual(out, cout)

    def test_array_ndenumerate_2d(self):
        arr = np.arange(12).reshape(4, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, "C")
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, "F")
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, "A")
        self.check_array_ndenumerate_sum(np.ascontiguousarray(arr), arrty)
        # Boolean array
        arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
        self.check_array_ndenumerate_sum(np.ascontiguousarray(arr), typeof(arr))

    def test_array_ndenumerate_empty(self):
        # Define a local checking function, Numba's `typeof` ends up aliasing
        # 0d C and F ordered arrays, so the check needs to go via the compile
        # result entry point to bypass type checking.
        def check(arr, arrty):
            out = np.zeros(1, dtype=np.int32)
            cout = np.zeros(1, dtype=np.int32)
            cfunc = jit((arrty, typeof(out)))(array_ndenumerate_sum)
            array_ndenumerate_sum(arr, out)
            cfunc[1, 1](arr, cout)
            np.testing.assert_allclose(out[0], cout[0])

        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout="C")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="F")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="A")
        check(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout="C")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="F")
        check(arr, arrty)
        arrty = types.Array(types.int32, 2, layout="A")
        check(arr, arrty)

    def test_array_ndenumerate_premature_free(self):
        with override_config("CUDA_ENABLE_NRT", True):
            out = np.zeros(6)
            cout = np.zeros(6)
            cfunc = jit((types.intp, typeof(out)))(
                array_ndenumerate_premature_free
            )
            array_ndenumerate_premature_free(6, out)
            cfunc[1, 1](6, cout)
            self.assertTrue(cout.sum())
            self.assertPreciseEqual(out, cout)

    def test_np_ndindex(self):
        func = np_ndindex
        out = np.zeros(1)
        cout = np.zeros(1)
        cfunc = jit(
            (
                types.int32,
                types.int32,
                typeof(out),
            )
        )(func)
        func(3, 4, out)
        cfunc[1, 1](3, 4, cout)
        self.assertPreciseEqual(out, cout)
        func(3, 0, out)
        cfunc[1, 1](3, 0, cout)
        self.assertPreciseEqual(out, cout)
        func(0, 3, out)
        cfunc[1, 1](0, 3, cout)
        self.assertPreciseEqual(out, cout)
        func(0, 0, out)
        cfunc[1, 1](0, 0, cout)
        self.assertPreciseEqual(out, cout)

    def test_np_ndindex_array(self):
        func = np_ndindex_array
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((4, 3))
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((2, 2, 3))
        self.check_array_unary(arr, typeof(arr), func)

    def test_iter_next(self):
        # This also checks memory management with iter() and next()
        func = iter_next
        arr = np.arange(12, dtype=np.int32) + 10
        self.check_array_unary(arr, typeof(arr), func)


class TestNdIter(MemoryLeakMixin, CUDATestCase):
    """
    Test np.nditer()
    """

    def inputs_a(self):
        # scalars
        yield np.float32(100)

        # 0-d arrays
        yield np.array(102, dtype=np.int16)

    def inputs_b(self):
        # All those inputs are compatible with a (3, 4) main shape
        # 1-d arrays
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]

        # 2-d arrays
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order="F")
        a = np.arange(24).reshape((6, 4))[::2]
        yield a

    def basic_inputs(self):
        yield np.arange(4).astype(np.complex64)
        yield np.arange(8)[::2]
        a = np.arange(12).reshape((3, 4))
        yield a
        yield a.copy(order="F")

    def check_result(self, got, expected):
        self.assertEqual(set(got), set(expected), (got, expected))

    def test_nditer1a(self):
        pyfunc = np_nditer1a
        cfunc = jit(pyfunc)
        for a in self.inputs_a():
            out = np.zeros(a.size, dtype=a.dtype)
            cout = np.zeros(a.size, dtype=a.dtype)
            pyfunc(a, out)
            cfunc[1, 1](a, cout)
            self.assertPreciseEqual(out, cout)

    def test_nditer1b(self):
        pyfunc = np_nditer1b
        cfunc = jit(pyfunc)
        for a in self.inputs_b():
            out = np.zeros(a.size, dtype=a.dtype)
            cout = np.zeros(a.size, dtype=a.dtype)
            pyfunc(np.ascontiguousarray(a), out)
            cfunc[1, 1](np.ascontiguousarray(a), cout)
            self.assertPreciseEqual(out, cout)

    def test_nditer2a(self):
        pyfunc = np_nditer2a
        cfunc = jit(pyfunc)
        for a, b in itertools.product(self.inputs_a(), self.inputs_a()):
            out = np.zeros(
                a.size + b.size, dtype=np.result_type(a.dtype, b.dtype)
            )
            cout = np.zeros(
                a.size + b.size, dtype=np.result_type(a.dtype, b.dtype)
            )
            pyfunc(a, b, out)
            cfunc[1, 1](a, b, cout)
            self.check_result(out, cout)

    def test_nditer2b(self):
        pyfunc = np_nditer2b
        cfunc = jit(pyfunc)
        for a, b in itertools.product(self.inputs_b(), self.inputs_b()):
            out = np.zeros(
                a.size * b.size, dtype=np.result_type(a.dtype, b.dtype)
            )
            cout = np.zeros(
                a.size * b.size, dtype=np.result_type(a.dtype, b.dtype)
            )
            pyfunc(np.ascontiguousarray(a), np.ascontiguousarray(b), out)
            cfunc[1, 1](np.ascontiguousarray(a), np.ascontiguousarray(b), cout)
            self.check_result(out, cout)

    def test_nditer3(self):
        pyfunc = np_nditer3
        cfunc = jit(pyfunc)
        # Use a restricted set of inputs, to shorten test time
        inputs = self.basic_inputs
        for a, b, c in itertools.product(inputs(), inputs(), inputs()):
            out = np.zeros(
                a.size * b.size * c.size,
                dtype=np.result_type(a.dtype, b.dtype, c.dtype),
            )
            cout = np.zeros(
                a.size * b.size * c.size,
                dtype=np.result_type(a.dtype, b.dtype, c.dtype),
            )
            pyfunc(
                np.ascontiguousarray(a),
                np.ascontiguousarray(b),
                np.ascontiguousarray(c),
                out,
            )
            cfunc[1, 1](
                np.ascontiguousarray(a),
                np.ascontiguousarray(b),
                np.ascontiguousarray(c),
                cout,
            )
            self.check_result(out, cout)

    def test_errors(self):
        # Incompatible shapes
        pyfunc = np_nditer2b_err
        cfunc = jit(debug=True, opt=False)(pyfunc)

        self.disable_leak_check()

        def check_incompatible(a, b):
            with self.assertRaises(SystemError) as raises:
                out = np.zeros(
                    a.size * b.size, dtype=np.result_type(a.dtype, b.dtype)
                )
                cfunc[1, 1](a, b, out)
            self.assertIn(
                "unknown error",
                str(raises.exception),
            )

        check_incompatible(np.arange(2), np.arange(3))
        a = np.arange(12).reshape((3, 4))
        b = np.arange(3)
        check_incompatible(a, b)


if __name__ == "__main__":
    unittest.main()
