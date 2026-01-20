import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def index_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        out[i] = inp[i]


@cuda.jit
def slice_kernel(inp, out, start):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[start + i]


@cuda.jit
def reshape_kernel(inp, out, width):
    i = cuda.grid(1)
    if i < out.size:
        row = i // width
        col = i % width
        out[row, col] = inp[i]


@skip_on_cudasim("Array slicing semantics differ under cudasim")
class TestCudaArrayManipulation(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()


    def test_basic_indexing(self):
        src = np.arange(10, dtype=np.int32)
        dst = np.zeros_like(src)

        d_src = cuda.to_device(src)
        d_dst = cuda.to_device(dst)

        self._launch_1d(index_kernel, (d_src, d_dst), src.size)

        np.testing.assert_array_equal(
            d_dst.copy_to_host(),
            src
        )


    def test_basic_slicing(self):
        src = np.arange(20, dtype=np.int32)
        start = 5
        length = 10

        expected = src[start:start + length]
        out = np.zeros(length, dtype=np.int32)

        d_src = cuda.to_device(src)
        d_out = cuda.to_device(out)

        self._launch_1d(slice_kernel, (d_src, d_out, start), length)

        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            expected
        )


    def test_simple_reshape(self):
        src = np.arange(12, dtype=np.int32)
        reshaped = src.reshape(3, 4)

        out = np.zeros_like(reshaped)

        d_src = cuda.to_device(src)
        d_out = cuda.to_device(out)

        self._launch_1d(
            reshape_kernel,
            (d_src, d_out, reshaped.shape[1]),
            src.size,
        )

        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            reshaped
        )
