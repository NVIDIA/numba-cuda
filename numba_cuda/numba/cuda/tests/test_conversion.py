import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def convert_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = inp[i]


@skip_on_cudasim("Conversion semantics differ under cudasim")
class TestCudaConversion(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()


    def _run_conversion(self, src, dst_dtype):
        out = np.zeros_like(src, dtype=dst_dtype)

        d_src = cuda.to_device(src)
        d_out = cuda.to_device(out)

        self._launch_1d(convert_kernel, (d_src, d_out), src.size)

        result = d_out.copy_to_host()
        expected = src.astype(dst_dtype)

        np.testing.assert_array_equal(result, expected)


    def test_int_to_float(self):
        src = np.array([1, -2, 3, 4], dtype=np.int32)
        self._run_conversion(src, np.float32)


    def test_float_to_int_truncation(self):
        src = np.array([1.7, -2.2, 3.9], dtype=np.float32)
        self._run_conversion(src, np.int32)


    def test_int_to_complex(self):
        src = np.array([1, -2, 3], dtype=np.int32)
        self._run_conversion(src, np.complex64)


    def test_float_to_complex(self):
        src = np.array([1.5, -2.25, 3.75], dtype=np.float32)
        self._run_conversion(src, np.complex64)


    def test_bool_to_int(self):
        src = np.array([True, False, True], dtype=np.bool_)
        self._run_conversion(src, np.int32)


    def test_int_to_bool(self):
        src = np.array([0, 1, -3, 4], dtype=np.int32)
        self._run_conversion(src, np.bool_)


    def test_bool_to_float(self):
        src = np.array([True, False, True], dtype=np.bool_)
        self._run_conversion(src, np.float32)


    def test_float_to_bool(self):
        src = np.array([0.0, 1.2, -0.5], dtype=np.float32)
        self._run_conversion(src, np.bool_)
