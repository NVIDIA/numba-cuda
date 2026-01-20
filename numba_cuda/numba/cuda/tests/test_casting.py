import numpy as np
import unittest

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def cast_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = inp[i]


class TestCudaCasting(CUDATestCase):

    def _run_cast_test(self, src_dtype, dst_dtype, values):
        src = np.array(values, dtype=src_dtype)
        dst = np.zeros_like(src, dtype=dst_dtype)

        d_src = cuda.to_device(src)
        d_dst = cuda.to_device(dst)

        threadsperblock = 128
        blockspergrid = (src.size + threadsperblock - 1) // threadsperblock

        cast_kernel[blockspergrid, threadsperblock](d_src, d_dst)

        result = d_dst.copy_to_host()
        expected = src.astype(dst_dtype)

        np.testing.assert_array_equal(result, expected)


    def test_int32_to_int64(self):
        self._run_cast_test(np.int32, np.int64, [1, 2, -3, 4])


    def test_int64_to_int32(self):
        self._run_cast_test(np.int64, np.int32, [1, 2, -3, 4])


    def test_int_to_float(self):
        self._run_cast_test(np.int32, np.float32, [1, -2, 3, 4])


    def test_float_to_int(self):
        self._run_cast_test(np.float32, np.int32, [1.7, -2.2, 3.9])


    def test_float32_to_float64(self):
        self._run_cast_test(np.float32, np.float64, [1.5, -2.5, 3.25])


    def test_bool_to_int(self):
        self._run_cast_test(np.bool_, np.int32, [True, False, True])


    def test_int_to_bool(self):
        self._run_cast_test(np.int32, np.bool_, [0, 1, 2, -1])
