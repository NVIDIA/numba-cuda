import numpy as np

from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba import cuda
from numba.core.errors import TypingError


def array_return(a):
    return a


def array_slice(a, start, end):
    return a[start:end]


def array_slice_2d(a, x_id, x_size, y_id, y_size):
    return a[
        x_id * x_size : (x_id + 1) * x_size : 1,
        y_id * y_size : (y_id + 1) * y_size : 1,
    ]


def array_local(shape, dtype):
    return cuda.local.array(shape, dtype=dtype)


class TestCudaArrayReturn(CUDATestCase):
    def test_array_return(self):
        f = cuda.jit(device=True)(array_return)

        @cuda.jit
        def kernel(x):
            f(x)

        a = np.zeros(5)

        kernel[1, 1](a)

    def test_array_slice(self):
        f = cuda.jit(device=True)(array_slice)

        @cuda.jit
        def kernel(x):
            y = f(x, 2, 3)
            y[0] = 1

        a = np.zeros(5)

        kernel[1, 1](a)

        assert a[0] == 0
        assert a[2] == 1

    def test_array_slice_2d(self):
        f = cuda.jit(device=True)(array_slice_2d)

        @cuda.jit
        def kernel(x):
            y = f(x, 1, 2, 2, 2)
            y[0, 0] = 1

        a = np.zeros((4, 6))

        kernel[1, 1](a)

        assert a[0, 0] == 0
        assert a[2, 4] == 1

    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_local(self):
        f = cuda.jit(device=True)(array_local)

        @cuda.jit
        def kernel():
            x = f((2, 3), np.int32)
            x[0, 0] = 1

        with self.assertRaises(TypingError):
            kernel[1, 1]()


if __name__ == "__main__":
    unittest.main()
