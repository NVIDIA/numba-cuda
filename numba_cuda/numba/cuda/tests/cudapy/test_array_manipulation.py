"""
CUDA array manipulation tests using kernels and Python CUDA semantics.

These tests validate array operations, indexing, and element-wise operations
in CUDA kernels. They are inspired by CPU array manipulation test suites and
adapted to verify CUDA device semantics.
"""

import numpy as np
import pytest

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def fill_kernel(arr, value):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = value


@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


class TestArrayManipulation(CUDATestCase):
    """Tests for array manipulation operations in CUDA kernels."""

    def test_fill_basic(self):
        """Basic elementwise fill operation - works in simulator."""
        n = 128
        arr = cuda.device_array(n, dtype=np.float32)

        threads = 64
        blocks = (n + threads - 1) // threads
        fill_kernel[blocks, threads](arr, 4.25)

        result = arr.copy_to_host()
        self.assertTrue(np.all(result == 4.25))

    def test_elementwise_add(self):
        """Elementwise addition - basic arithmetic works in simulator."""
        n = 256
        a = np.arange(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.device_array_like(a)

        threads = 64
        blocks = (n + threads - 1) // threads
        add_kernel[blocks, threads](da, db, dout)

        result = dout.copy_to_host()
        self.assertTrue(np.allclose(result, a + b))

    def test_integer_getitem_setitem(self):
        """Direct integer indexing in kernel - works in simulator."""
        arr = cuda.device_array(10, dtype=np.int32)

        @cuda.jit
        def set_index(a):
            a[3] = 42

        set_index[1, 1](arr)
        host = arr.copy_to_host()
        self.assertEqual(host[3], 42)

    def test_multidimensional_indexing(self):
        """2D indexing with grid-based kernel - works in simulator."""
        shape = (8, 8)
        host = np.zeros(shape, dtype=np.int32)
        dev = cuda.to_device(host)

        @cuda.jit
        def write_2d(a):
            x, y = cuda.grid(2)
            if x < a.shape[0] and y < a.shape[1]:
                a[x, y] = x * a.shape[1] + y

        threads = (4, 4)
        blocks = ((shape[0] + threads[0] - 1) // threads[0],
                  (shape[1] + threads[1] - 1) // threads[1])

        write_2d[blocks, threads](dev)
        result = dev.copy_to_host()
        self.assertEqual(result[2, 3], 2 * shape[1] + 3)

    @pytest.mark.xfail(reason="Advanced slicing not supported on CUDA device arrays")
    def test_advanced_slicing(self):
        """Advanced slicing (e.g., ::2) is not supported on device arrays."""
        arr = cuda.to_device(np.arange(10))
        _ = arr[::2]

    @pytest.mark.xfail(reason="Multidimensional slicing not supported on CUDA device arrays")
    def test_multidimensional_slicing(self):
        """Multidimensional slicing not supported on device arrays."""
        arr = cuda.to_device(np.arange(16).reshape(4, 4))
        _ = arr[:, ::-1]

    @pytest.mark.xfail(reason="Boolean indexing not supported on CUDA device arrays")
    def test_boolean_indexing(self):
        """Boolean mask indexing is not supported on device arrays."""
        arr = cuda.to_device(np.arange(10))
        mask = np.array([True, False] * 5)
        _ = arr[mask]

    @pytest.mark.xfail(reason="Fancy indexing with arrays not supported on CUDA device arrays")
    def test_fancy_indexing(self):
        """Fancy indexing with integer arrays not supported on device arrays."""
        arr = cuda.to_device(np.arange(10))
        idx = np.array([1, 3, 5])
        _ = arr[idx]

    def test_dtype_transitions(self):
        """Type casting and mixed-dtype operations - basic ops work in simulator."""
        n = 10
        a = np.arange(n, dtype=np.int32)
        b = np.arange(n, dtype=np.float32)
        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout1 = cuda.device_array(n, dtype=np.float32)
        dout2 = cuda.device_array(n, dtype=np.int32)

        @cuda.jit
        def cast_int_to_float(src, dst):
            i = cuda.grid(1)
            if i < src.size:
                dst[i] = src[i]

        @cuda.jit
        def cast_float_to_int(src, dst):
            i = cuda.grid(1)
            if i < src.size:
                dst[i] = int(src[i])

        threads = 32
        blocks = (n + threads - 1) // threads
        cast_int_to_float[blocks, threads](da, dout1)
        cast_float_to_int[blocks, threads](db, dout2)

        self.assertTrue(np.allclose(dout1.copy_to_host(), a.astype(np.float32)))
        self.assertTrue(np.all(dout2.copy_to_host() == b.astype(np.int32)))

        # Mixed dtype elementwise op
        dout3 = cuda.device_array(n, dtype=np.float32)

        @cuda.jit
        def add_mixed(a, b, out):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = a[i] + b[i]

        add_mixed[blocks, threads](da, db, dout3)
        self.assertTrue(np.allclose(dout3.copy_to_host(), a + b))

    def test_shape_semantics_in_kernel(self):
        """Shape attributes accessible in kernel - works in simulator."""
        shape = (5, 7)
        arr = cuda.device_array(shape, dtype=np.int32)

        @cuda.jit
        def check_shape(a, out):
            if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
                out[0] = a.shape[0]
                out[1] = a.shape[1]

        out = cuda.device_array(2, dtype=np.int32)
        check_shape[1, 1](arr, out)
        host = out.copy_to_host()
        self.assertEqual(tuple(host), shape)

    @pytest.mark.xfail(reason="reshape not supported on CUDA device arrays")
    def test_reshape(self):
        """reshape is not supported on device arrays."""
        arr = cuda.device_array((4, 4), dtype=np.int32)
        _ = arr.reshape((16,))

    @pytest.mark.xfail(reason="view not supported on CUDA device arrays")
    def test_view(self):
        """view is not supported on device arrays."""
        arr = cuda.device_array(8, dtype=np.int32)
        _ = arr.view(np.float32)

    @pytest.mark.xfail(reason="ravel not supported on CUDA device arrays")
    def test_ravel(self):
        """ravel is not supported on device arrays."""
        arr = cuda.device_array((2, 4), dtype=np.int32)
        _ = arr.ravel()

    @skip_on_cudasim("True grid-stride loop semantics require real device execution")
    def test_grid_stride_correctness(self):
        """Grid-stride loop pattern - requires actual CUDA grid execution."""
        n = 10
        arr = cuda.device_array(n, dtype=np.int32)

        @cuda.jit
        def grid_stride_kernel(a):
            for i in range(cuda.grid(1), a.size, cuda.gridsize(1)):
                a[i] = i

        threads = 32  # threads > n
        blocks = 2    # blocks > needed
        grid_stride_kernel[blocks, threads](arr)
        host = arr.copy_to_host()
        self.assertTrue(np.all(host == np.arange(n)))

    def test_cpu_parity_reference(self):
        """Validates CUDA behavior matches expected CPU NumPy semantics."""
        n = 50
        a = np.arange(n, dtype=np.float32)
        b = np.arange(n, dtype=np.float32)
        expected = a + b
        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.device_array_like(a)

        threads = 16
        blocks = (n + threads - 1) // threads
        add_kernel[blocks, threads](da, db, dout)
        result = dout.copy_to_host()
        self.assertTrue(np.allclose(result, expected))
