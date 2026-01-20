import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim

# -----------------------------
# Predicate kernels
# -----------------------------

@cuda.jit
def greater_than_zero_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] > 0:
            out[i] = 1
        else:
            out[i] = 0

@cuda.jit
def equality_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] == 3:
            out[i] = 10
        else:
            out[i] = -1

@cuda.jit
def bool_array_kernel(mask, out):
    i = cuda.grid(1)
    if i < out.size:
        if mask[i]:
            out[i] = 1
        else:
            out[i] = 0

@cuda.jit
def nested_predicate_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] > 0:
            if inp[i] % 2 == 0:
                out[i] = 2
            else:
                out[i] = 1
        else:
            out[i] = 0

@cuda.jit
def constant_predicate_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        if True:
            out[i] = 7
        else:
            out[i] = 0

@skip_on_cudasim("Predicate lowering differs under cudasim")
class TestCudaConditionsAsPredicates(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()

    def test_greater_than_zero(self):
        inp = np.array([-2, -1, 0, 1, 3], dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(greater_than_zero_kernel, (d_inp, d_out), inp.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            (inp > 0).astype(np.int32),
        )

    def test_equality_predicate(self):
        inp = np.array([1, 3, 5, 3, 0], dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(equality_kernel, (d_inp, d_out), inp.size)
        expected = np.where(inp == 3, 10, -1)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            expected,
        )

    def test_bool_array_predicate(self):
        mask = np.array([True, False, True, False], dtype=np.bool_)
        out = np.zeros(mask.size, dtype=np.int32)
        d_mask = cuda.to_device(mask)
        d_out = cuda.to_device(out)
        self._launch_1d(bool_array_kernel, (d_mask, d_out), mask.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            mask.astype(np.int32),
        )

    def test_nested_predicates(self):
        inp = np.array([-2, -1, 1, 2, 3, 4], dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(nested_predicate_kernel, (d_inp, d_out), inp.size)
        expected = np.zeros_like(inp)
        for i, v in enumerate(inp):
            if v > 0:
                expected[i] = 2 if v % 2 == 0 else 1
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            expected,
        )

    def test_constant_predicate(self):
        out = np.zeros(6, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(constant_predicate_kernel, (d_out,), out.size)
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            np.full_like(out, 7),
        )
