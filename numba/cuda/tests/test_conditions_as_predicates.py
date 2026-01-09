import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def if_scalar_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i]:
            out[i] = 1
        else:
            out[i] = 0


@cuda.jit
def if_comparison_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] > 0:
            out[i] = 1
        else:
            out[i] = 0


@cuda.jit
def if_float_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        # Explicit float predicate
        if inp[i] > 0.5:
            out[i] = 1
        else:
            out[i] = 0


@cuda.jit
def nested_if_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if inp[i] > 0:
            if inp[i] > 10:
                out[i] = 2
            else:
                out[i] = 1
        else:
            out[i] = 0


@cuda.jit
def logical_and_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        if a[i] and b[i]:
            out[i] = 1
        else:
            out[i] = 0


@cuda.jit
def logical_or_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        if a[i] or b[i]:
            out[i] = 1
        else:
            out[i] = 0


class TestConditionsAsPredicates(CUDATestCase):

    def _launch_1d(self, kernel, *args: np.ndarray) -> None:
        size = args[0].size
        threadsperblock = 64
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)

    @skip_on_cudasim("Predicate lowering requires device execution")
    def test_if_on_scalar_int(self):
        inp = np.arange(-64, 64, dtype=np.int32)
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(if_scalar_kernel, d_inp, d_out)

        expected = (inp != 0).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Predicate lowering requires device execution")
    def test_if_on_comparison(self):
        inp = np.arange(-64, 64, dtype=np.int32)
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(if_comparison_kernel, d_inp, d_out)

        expected = (inp > 0).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Float predicate lowering requires device execution")
    def test_if_on_float(self):
        inp = np.array(
            [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, np.inf],
            dtype=np.float32,
        )
        inp = np.repeat(inp, 32)
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(if_float_kernel, d_inp, d_out)

        expected = (inp > 0.5).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Nested predicate lowering requires device execution")
    def test_nested_if(self):
        inp = np.array([-5, 0, 3, 10, 11, 20], dtype=np.int32)
        inp = np.repeat(inp, 32)
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(nested_if_kernel, d_inp, d_out)

        expected = np.zeros_like(inp)
        expected[inp > 0] = 1
        expected[inp > 10] = 2

        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Logical AND requires device execution")
    def test_logical_and(self):
        a = np.array([0, 0, 1, 1], dtype=np.int32)
        b = np.array([0, 1, 0, 1], dtype=np.int32)

        a = np.repeat(a, 32)
        b = np.repeat(b, 32)
        out = np.zeros(a.size, dtype=np.int32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.to_device(out)

        self._launch_1d(logical_and_kernel, d_a, d_b, d_out)

        expected = (a.astype(bool) & b.astype(bool)).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Logical OR requires device execution")
    def test_logical_or(self):
        a = np.array([0, 0, 1, 1], dtype=np.int32)
        b = np.array([0, 1, 0, 1], dtype=np.int32)

        a = np.repeat(a, 32)
        b = np.repeat(b, 32)
        out = np.zeros(a.size, dtype=np.int32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.to_device(out)

        self._launch_1d(logical_or_kernel, d_a, d_b, d_out)

        expected = (a.astype(bool) | b.astype(bool)).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

    @skip_on_cudasim("Explicit warp divergence requires device execution")
    def test_explicit_warp_divergence(self):
        # Alternate truth values within a single warp to force divergence
        inp = np.arange(64, dtype=np.int32) % 2
        out = np.zeros(inp.size, dtype=np.int32)

        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)

        self._launch_1d(if_scalar_kernel, d_inp, d_out)

        expected = (inp != 0).astype(np.int32)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)

