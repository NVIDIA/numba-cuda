import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim

# -----------------------------
# Kernels using auto-constants
# -----------------------------

@cuda.jit
def int_constant_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = 42

@cuda.jit
def float_constant_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = 3.5

@cuda.jit
def bool_constant_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = True

@cuda.jit
def arithmetic_constant_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = 2 + 3 * 4  # 14

@cuda.jit
def mixed_constant_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = 1.5 + 2   # 3.5

@cuda.jit
def constant_index_kernel(inp, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[2]

@skip_on_cudasim("Auto-constant lowering differs under cudasim")
class TestCudaAutoConstants(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()

    def test_int_constant(self):
        out = np.zeros(8, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(int_constant_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 42),
        )

    def test_float_constant(self):
        out = np.zeros(8, dtype=np.float32)
        d_out = cuda.to_device(out)
        self._launch_1d(float_constant_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 3.5),
        )

    def test_bool_constant(self):
        out = np.zeros(8, dtype=np.bool_)
        d_out = cuda.to_device(out)
        self._launch_1d(bool_constant_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.ones_like(out, dtype=np.bool_),
        )

    def test_arithmetic_constant(self):
        out = np.zeros(8, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(arithmetic_constant_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 14),
        )

    def test_mixed_constant(self):
        out = np.zeros(8, dtype=np.float32)
        d_out = cuda.to_device(out)
        self._launch_1d(mixed_constant_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 3.5),
        )

    def test_constant_indexing(self):
        inp = np.arange(10, dtype=np.int32)
        out = np.zeros(5, dtype=np.int32)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(constant_index_kernel, (d_inp, d_out), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, inp[2]),
        )
