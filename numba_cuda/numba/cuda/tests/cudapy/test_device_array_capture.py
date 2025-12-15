# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Tests for capturing device arrays (objects implementing __cuda_array_interface__)
from global scope in CUDA kernels and device functions.

This tests the capture of third-party arrays (like CuPy) that implement
__cuda_array_interface__ but don't have Numba's _numba_type_ attribute.
"""

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim

# Check if CuPy is available
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

skip_unless_cupy = unittest.skipUnless(
    HAS_CUPY, "CuPy is required for this test"
)


# =============================================================================
# Global arrays for testing - using CuPy arrays
# =============================================================================

# Create global arrays that will be captured (initialized in test setup)
_global_array_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
_global_cupy_array = None  # Will be initialized in test setup


def _create_global_arrays():
    """Create the global arrays. Called during test setup."""
    global _global_cupy_array
    _global_cupy_array = cp.asarray(_global_array_data)


def _cleanup_global_arrays():
    """Clean up the global arrays. Called during test teardown."""
    global _global_cupy_array
    _global_cupy_array = None


@skip_on_cudasim("Global device array capture not supported in simulator")
@skip_unless_cupy
class TestDeviceArrayCapture(CUDATestCase):
    """Test capturing CuPy arrays from global scope."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        _create_global_arrays()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        _cleanup_global_arrays()

    def test_basic_global_capture(self):
        """
        Test that a device function can reference a global CuPy array
        and a kernel can call that function.
        """

        # Define a device function that reads from the global array
        @cuda.jit(device=True)
        def read_global(idx):
            return _global_cupy_array[idx]

        # Define a kernel that calls the device function
        @cuda.jit
        def kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = read_global(i)

        # Run the kernel
        n = len(_global_array_data)
        output = cuda.device_array(n, dtype=np.float32)
        kernel[1, n](output)

        # Validate results
        result = output.copy_to_host()
        np.testing.assert_array_equal(result, _global_array_data)

    def test_global_capture_with_computation(self):
        """
        Test that captured global arrays can be used in computations.
        """

        @cuda.jit(device=True)
        def double_global_value(idx):
            return _global_cupy_array[idx] * 2.0

        @cuda.jit
        def kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = double_global_value(i)

        n = len(_global_array_data)
        output = cuda.device_array(n, dtype=np.float32)
        kernel[1, n](output)

        result = output.copy_to_host()
        expected = _global_array_data * 2.0
        np.testing.assert_array_equal(result, expected)

    def test_mutability_of_captured_array(self):
        """
        Test that captured CuPy arrays can be written to (mutability).
        """
        # Create a separate mutable CuPy array for this test
        mutable_array = cp.zeros(5, dtype=np.float32)

        @cuda.jit
        def write_kernel():
            i = cuda.grid(1)
            if i < 5:
                mutable_array[i] = float(i + 1)

        write_kernel[1, 5]()

        # Copy back using CuPy's get method
        result = cp.asnumpy(mutable_array)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_global_arrays(self):
        """
        Test capturing multiple CuPy arrays from globals.
        """
        cupy_a = cp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        cupy_b = cp.asarray(np.array([10.0, 20.0, 30.0], dtype=np.float32))

        @cuda.jit(device=True)
        def add_globals(idx):
            return cupy_a[idx] + cupy_b[idx]

        @cuda.jit
        def kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = add_globals(i)

        output = cuda.device_array(3, dtype=np.float32)
        kernel[1, 3](output)

        result = output.copy_to_host()
        expected = np.array([11.0, 22.0, 33.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional_array_capture(self):
        """
        Test capturing multidimensional CuPy arrays.
        """
        host_2d = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        cupy_2d = cp.asarray(host_2d)

        @cuda.jit(device=True)
        def read_2d(row, col):
            return cupy_2d[row, col]

        @cuda.jit
        def kernel(output):
            i = cuda.grid(1)
            if i < 6:
                row = i // 2
                col = i % 2
                output[i] = read_2d(row, col)

        output = cuda.device_array(6, dtype=np.float32)
        kernel[1, 6](output)

        result = output.copy_to_host()
        expected = host_2d.flatten()
        np.testing.assert_array_equal(result, expected)

    def test_different_dtypes(self):
        """
        Test capturing CuPy arrays with different data types.
        """
        # Integer array
        int_host = np.array([10, 20, 30, 40], dtype=np.int32)
        int_cupy = cp.asarray(int_host)

        @cuda.jit(device=True)
        def read_int(idx):
            return int_cupy[idx]

        @cuda.jit
        def int_kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = read_int(i)

        int_output = cuda.device_array(4, dtype=np.int32)
        int_kernel[1, 4](int_output)
        np.testing.assert_array_equal(int_output.copy_to_host(), int_host)

        # Float64 array
        f64_host = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        f64_cupy = cp.asarray(f64_host)

        @cuda.jit(device=True)
        def read_f64(idx):
            return f64_cupy[idx]

        @cuda.jit
        def f64_kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = read_f64(i)

        f64_output = cuda.device_array(3, dtype=np.float64)
        f64_kernel[1, 3](f64_output)
        np.testing.assert_array_equal(f64_output.copy_to_host(), f64_host)

    def test_direct_kernel_global_access(self):
        """
        Test that kernels can directly access global CuPy arrays
        (not just through device functions).
        """
        global_direct = cp.asarray(np.array([7.0, 8.0, 9.0], dtype=np.float32))

        @cuda.jit
        def direct_access_kernel(output):
            i = cuda.grid(1)
            if i < output.size:
                output[i] = global_direct[i] + 1.0

        output = cuda.device_array(3, dtype=np.float32)
        direct_access_kernel[1, 3](output)

        result = output.copy_to_host()
        expected = np.array([8.0, 9.0, 10.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


@skip_on_cudasim("Global device array capture not supported in simulator")
@skip_unless_cupy
class TestDeviceArrayCaptureCaching(CUDATestCase):
    """
    Test that caching correctly rejects kernels with captured device arrays.

    When kernels are compiled with cache=True, the device pointer would be
    embedded in the cached code. This is problematic because:
    1. The original device array may have been deallocated
    2. A new device array may be allocated at a different address

    Numba's serialization correctly rejects pickling of objects implementing
    __cuda_array_interface__, preventing this issue.
    """

    def test_caching_rejects_captured_pointer(self):
        """
        Test that caching is correctly rejected for kernels with captured
        CuPy arrays (which contain device pointers).
        """
        import pickle
        import tempfile
        import shutil
        import os

        # Create a temporary cache directory
        cache_dir = tempfile.mkdtemp(prefix="numba_cache_test_")

        try:
            # Set up the cache directory
            old_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
            os.environ["NUMBA_CACHE_DIR"] = cache_dir

            # Create CuPy array
            host_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            cupy_arr = cp.asarray(host_data)

            # Define a cached kernel that captures the CuPy array
            @cuda.jit(cache=True)
            def cached_kernel(output):
                i = cuda.grid(1)
                if i < output.size:
                    output[i] = cupy_arr[i] * 2.0

            output = cuda.device_array(3, dtype=np.float32)

            # This should raise a PicklingError because CAI objects contain
            # device pointers that cannot be safely serialized for caching
            with self.assertRaises(pickle.PicklingError) as cm:
                cached_kernel[1, 3](output)

            # Verify the error message is helpful
            self.assertIn("__cuda_array_interface__", str(cm.exception))
            self.assertIn("cache=False", str(cm.exception))

        finally:
            # Restore environment
            if old_cache_dir is not None:
                os.environ["NUMBA_CACHE_DIR"] = old_cache_dir
            elif "NUMBA_CACHE_DIR" in os.environ:
                del os.environ["NUMBA_CACHE_DIR"]

            # Clean up cache directory
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
