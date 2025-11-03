# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import unittest
from numba.cuda.np.numpy_support import from_dtype
from numba import typeof
from numba.cuda import jit
from numba.core import types
from numba.cuda.tests.support import MemoryLeakMixin
from numba.cuda.testing import CUDATestCase
from numba.cuda.core.errors import TypingError
from numba.cuda.tests.support import override_config
from numba.cuda import config

if config.ENABLE_CUDASIM:
    raise unittest.SkipTest("Array attribute tests not done in simulator")


def array_itemsize(a, res):
    res[0] = a.itemsize


def array_nbytes(a, res):
    res[0] = a.nbytes


def array_shape(a, i, res):
    res[0] = a.shape[i]


def array_strides(a, i, res):
    res[0] = a.strides[i]


def array_ndim(a, res):
    res[0] = a.ndim


def array_size(a, res):
    res[0] = a.size


def array_flags_contiguous(a, res):
    res[0] = a.flags.contiguous


def array_flags_c_contiguous(a, res):
    res[0] = a.flags.c_contiguous


def array_flags_f_contiguous(a, res):
    res[0] = a.flags.f_contiguous


def nested_array_itemsize(a, res):
    res[0] = a.f.itemsize


def nested_array_nbytes(a, res):
    res[0] = a.f.nbytes


def nested_array_shape(a, res):
    res[0] = a.f.shape[0]
    res[1] = a.f.shape[1]


def nested_array_strides(a, res):
    res[0] = a.f.strides[0]
    res[1] = a.f.strides[1]


def nested_array_ndim(a, res):
    res[0] = a.f.ndim


def nested_array_size(a, res):
    res[0] = a.f.size


def size_after_slicing_usecase(buf, i, res):
    sliced = buf[i]
    # Make sure size attribute is not lost
    res[0] = sliced.size


def array_real(arr, res):
    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            res[i] = arr.real[i]
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res[i, j] = arr.real[i, j]


def array_imag(arr, res):
    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            res[i] = arr.imag[i]
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                res[i, j] = arr.imag[i, j]


class TestArrayAttr(MemoryLeakMixin, CUDATestCase):
    def setUp(self):
        super(TestArrayAttr, self).setUp()
        self.a = np.arange(20, dtype=np.int32).reshape(4, 5)

    def check_unary(self, pyfunc, arr):
        out = np.zeros(1)
        aryty = typeof(arr)
        cfunc = self.get_cfunc(pyfunc, (aryty, typeof(out)))
        cout = np.zeros(1)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out[0], cout[0])
        # Retry with forced any layout
        cfunc = self.get_cfunc(pyfunc, (aryty.copy(layout="A"), typeof(out)))
        cout = np.zeros(1)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(cout[0], out[0])

    def check_unary_with_arrays(
        self,
        pyfunc,
    ):
        self.check_unary(pyfunc, self.a)
        self.check_unary(pyfunc, self.a.T)
        # 0-d array
        arr = np.array([42]).reshape(())
        self.check_unary(pyfunc, arr)
        # array with an empty dimension
        arr = np.zeros(0)
        self.check_unary(pyfunc, arr)

        # check with reshape
        self.check_unary(pyfunc, arr.reshape((1, 0, 2)))

    def get_cfunc(self, pyfunc, argspec):
        return jit(argspec)(pyfunc)

    def test_shape(self):
        pyfunc = array_shape
        cfunc = self.get_cfunc(
            pyfunc, (types.int32[:, :], types.int32, types.float64[:])
        )

        for i in range(self.a.ndim):
            out = np.zeros(1)
            cout = np.zeros(1)
            pyfunc(self.a, i, out)
            cfunc[1, 1](self.a, i, cout)
            self.assertEqual(out[0], cout[0])

    def test_strides(self):
        pyfunc = array_strides
        cfunc = self.get_cfunc(
            pyfunc, (types.int32[:, :], types.int32, types.float64[:])
        )

        for i in range(self.a.ndim):
            out = np.zeros(1)
            cout = np.zeros(1)
            pyfunc(self.a, i, out)
            cfunc[1, 1](self.a, i, cout)
            self.assertEqual(out[0], cout[0])

    def test_ndim(self):
        self.check_unary_with_arrays(array_ndim)

    def test_size(self):
        self.check_unary_with_arrays(array_size)

    def test_itemsize(self):
        self.check_unary_with_arrays(array_itemsize)

    def test_nbytes(self):
        self.check_unary_with_arrays(array_nbytes)

    def test_flags_contiguous(self):
        with override_config("CUDA_ENABLE_NRT", True):
            self.check_unary_with_arrays(array_flags_contiguous)

    def test_flags_c_contiguous(self):
        with override_config("CUDA_ENABLE_NRT", True):
            self.check_unary_with_arrays(array_flags_c_contiguous)

    def test_flags_f_contiguous(self):
        with override_config("CUDA_ENABLE_NRT", True):
            self.check_unary_with_arrays(array_flags_f_contiguous)


class TestNestedArrayAttr(MemoryLeakMixin, CUDATestCase):
    def setUp(self):
        super(TestNestedArrayAttr, self).setUp()
        dtype = np.dtype([("a", np.int32), ("f", np.int32, (2, 5))])
        self.a = np.recarray(1, dtype)[0]
        self.nbrecord = from_dtype(self.a.dtype)

    def get_cfunc(self, pyfunc):
        return jit((self.nbrecord, types.float64[:]))(pyfunc)

    def test_shape(self):
        pyfunc = nested_array_shape
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(2)
        cout = np.zeros(2)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])
        self.assertEqual(out[1], cout[1])

    def test_strides(self):
        pyfunc = nested_array_strides
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(2)
        cout = np.zeros(2)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])
        self.assertEqual(out[1], cout[1])

    def test_ndim(self):
        pyfunc = nested_array_ndim
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(1)
        cout = np.zeros(1)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])

    def test_nbytes(self):
        pyfunc = nested_array_nbytes
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(1)
        cout = np.zeros(1)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])

    def test_size(self):
        pyfunc = nested_array_size
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(1)
        cout = np.zeros(1)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])

    def test_itemsize(self):
        pyfunc = nested_array_itemsize
        cfunc = self.get_cfunc(pyfunc)

        out = np.zeros(1)
        cout = np.zeros(1)
        pyfunc(self.a, out)
        cfunc[1, 1](self.a, cout)
        self.assertEqual(out[0], cout[0])


class TestSlicedArrayAttr(MemoryLeakMixin, CUDATestCase):
    def test_size_after_slicing(self):
        pyfunc = size_after_slicing_usecase
        cfunc = jit(pyfunc)
        arr = np.arange(2 * 5).reshape(2, 5)
        for i in range(arr.shape[0]):
            out = np.zeros(1)
            cout = np.zeros(1)
            pyfunc(arr, i, out)
            cfunc[1, 1](arr, i, cout)
            self.assertEqual(out[0], cout[0])
        arr = np.arange(2 * 5 * 3).reshape(2, 5, 3)
        for i in range(arr.shape[0]):
            out = np.zeros(1)
            cout = np.zeros(1)
            pyfunc(arr, i, out)
            cfunc[1, 1](arr, i, cout)
            self.assertEqual(out[0], cout[0])


class TestRealImagAttr(MemoryLeakMixin, CUDATestCase):
    def setUp(self):
        override_config("CUDA_ENABLE_NRT", True)
        super(TestRealImagAttr, self).setUp()

    def check_complex(self, pyfunc):
        cfunc = jit(pyfunc)
        # test 1D
        size = 10
        arr = np.arange(size) + np.arange(size) * 10j
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)
        # test 2D
        arr = arr.reshape(2, 5)
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)

    def test_complex_real(self):
        self.check_complex(array_real)

    def test_complex_imag(self):
        self.check_complex(array_imag)

    def check_number_real(self, dtype):
        pyfunc = array_real
        cfunc = jit(pyfunc)
        # test 1D
        size = 10
        arr = np.arange(size, dtype=dtype)
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)
        # test 2D
        arr = arr.reshape(2, 5)
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)
        # test identity
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertEqual(arr.data, out.data)
        self.assertEqual(arr.data, cout.data)
        # test writable
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        cfunc[1, 1](arr, cout)
        self.assertNotEqual(cout[0, 0], 5)
        cout[0, 0] = 5
        self.assertEqual(cout[0, 0], 5)

    def test_number_real(self):
        """
        Testing .real of non-complex dtypes
        """
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            self.check_number_real(dtype)

    def check_number_imag(self, dtype):
        pyfunc = array_imag
        cfunc = jit(pyfunc)
        # test 1D
        size = 10
        arr = np.arange(size, dtype=dtype)
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)
        # test 2D
        arr = arr.reshape(2, 5)
        out = np.zeros(arr.shape)
        cout = np.zeros(arr.shape)
        pyfunc(arr, out)
        cfunc[1, 1](arr, cout)
        self.assertPreciseEqual(out, cout)
        # test are zeros
        cout = np.zeros(arr.shape)
        cfunc[1, 1](arr, cout)
        self.assertEqual(cout.tolist(), np.zeros_like(arr).tolist())

    def test_number_imag(self):
        """
        Testing .imag of non-complex dtypes
        """
        with override_config("CUDA_ENABLE_NRT", True):
            for dtype in [np.uint8, np.int32, np.float32, np.float64]:
                self.check_number_imag(dtype)

    def test_record_real(self):
        rectyp = np.dtype([("real", np.float32), ("imag", np.complex64)])
        arr = np.zeros(3, dtype=rectyp)
        arr["real"] = np.random.random(arr.size)
        arr["imag"] = np.random.random(arr.size) * 1.3j

        # check numpy behavior
        # .real is identity
        out = np.zeros(arr.shape, dtype=arr.dtype)
        array_real(arr, out)
        self.assertPreciseEqual(out, arr)
        # .imag is zero_like
        out = np.zeros(arr.shape, dtype=arr.dtype)
        array_imag(arr, out)
        self.assertEqual(out.tolist(), np.zeros_like(arr).tolist())

        # check numba behavior
        # it's most likely a user error, anyway
        jit_array_real = jit(array_real)
        jit_array_imag = jit(array_imag)

        cout = np.zeros(arr.shape, dtype=arr.dtype)
        with self.assertRaises(TypingError) as raises:
            jit_array_real[1, 1](arr, cout)
        self.assertIn(
            "cannot access .real of array of Record", str(raises.exception)
        )

        cout = np.zeros(arr.shape, dtype=arr.dtype)
        with self.assertRaises(TypingError) as raises:
            jit_array_imag[1, 1](arr, cout)
        self.assertIn(
            "cannot access .imag of array of Record", str(raises.exception)
        )


if __name__ == "__main__":
    unittest.main()
