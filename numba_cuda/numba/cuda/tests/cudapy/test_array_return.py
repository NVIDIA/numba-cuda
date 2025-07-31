import numpy as np
import pytest

from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba import cuda
from numba.core.errors import TypingError


class TestCudaArrayReturn(CUDATestCase):
    def test_array_return(self):
        @cuda.jit
        def array_return(a):
            return a

        @cuda.jit
        def kernel(x):
            y = array_return(x)
            y[2] = 1

        a = np.zeros(5)

        kernel[1, 1](a)

        assert a[0] == 0
        assert a[2] == 1

    def test_array_slice(self):
        @cuda.jit
        def array_slice(a, start, end):
            return a[start:end]

        @cuda.jit
        def kernel(x):
            y = array_slice(x, 2, 3)
            y[0] = 1

        a = np.zeros(5)

        kernel[1, 1](a)

        assert a[0] == 0
        assert a[2] == 1

    def test_array_slice_2d(self):
        @cuda.jit
        def array_slice_2d(a, x_id, x_size, y_id, y_size):
            return a[
                x_id * x_size : (x_id + 1) * x_size : 1,
                y_id * y_size : (y_id + 1) * y_size : 1,
            ]

        @cuda.jit
        def kernel(x):
            y = array_slice_2d(x, 1, 2, 2, 2)
            y[0, 0] = 1

        a = np.zeros((4, 6))

        kernel[1, 1](a)

        assert a[0, 0] == 0
        assert a[2, 4] == 1

    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_local_illegal(self):
        @cuda.jit
        def array_local(shape, dtype):
            return cuda.local.array(shape, dtype=dtype)

        @cuda.jit
        def kernel():
            x = array_local((2, 3), np.int32)
            x[0, 0] = 1

        with self.assertRaises(TypingError):
            kernel[1, 1]()

    @pytest.mark.xfail(reason="Returning const arrays is not yet supported")
    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_const(self):
        const_data = np.asarray([1, 2, 3])

        @cuda.jit
        def array_const():
            return cuda.const.array_like(const_data)

        r = np.zeros_like(const_data)

        @cuda.jit
        def kernel(r):
            x = array_const()
            i = cuda.grid(1)

            if i < len(r):
                r[i] = x[i]

        kernel[1, 1](r)

        np.testing.assert_equal(r, const_data)


if __name__ == "__main__":
    unittest.main()
