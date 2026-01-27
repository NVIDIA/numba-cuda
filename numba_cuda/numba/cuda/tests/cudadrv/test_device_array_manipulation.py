# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
DeviceArray behavior and memory management tests.

These tests validate DeviceArray shape, memory lifetime, and device memory
operations. Inspired by CPU array tests, these verify CUDA-specific device
array behaviors and constraints.
"""

import numpy as np
import pytest

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@cuda.jit
def set_val_kernel(a, v):
    i = cuda.grid(1)
    if i < a.size:
        a[i] = v


class TestDeviceArrayManipulation(CUDATestCase):
    """Tests for DeviceArray memory behavior and shape semantics."""

    @skip_on_cudasim("Device memory lifetime tracking requires real device memory")
    def test_memory_lifetime(self):
        """Verify device memory persists correctly across multiple kernel launches."""
        n = 32
        arr = cuda.device_array(n, dtype=np.float32)

        threads = 16
        blocks = (n + threads - 1) // threads
        set_val_kernel[blocks, threads](arr, 1.5)
        set_val_kernel[blocks, threads](arr, 2.5)
        host = arr.copy_to_host()
        self.assertTrue(np.allclose(host, 2.5))

    def test_device_array_shape(self):
        """Verify DeviceArray shape attribute is correct."""
        shape = (10, 20)
        arr = cuda.device_array(shape, dtype=np.float32)
        self.assertEqual(arr.shape, shape)

    def test_device_array_dtype(self):
        """Verify DeviceArray dtype attribute is correct."""
        dtype = np.int32
        arr = cuda.device_array(10, dtype=dtype)
        self.assertEqual(arr.dtype, dtype)

    def test_device_array_size(self):
        """Verify DeviceArray size attribute is correct."""
        shape = (5, 8)
        arr = cuda.device_array(shape, dtype=np.float32)
        self.assertEqual(arr.size, 40)

    def test_device_array_ndim(self):
        """Verify DeviceArray ndim attribute is correct."""
        arr1d = cuda.device_array(10, dtype=np.float32)
        arr2d = cuda.device_array((5, 8), dtype=np.float32)
        arr3d = cuda.device_array((2, 3, 4), dtype=np.float32)
        self.assertEqual(arr1d.ndim, 1)
        self.assertEqual(arr2d.ndim, 2)
        self.assertEqual(arr3d.ndim, 3)

    def test_device_array_like(self):
        """Verify device_array_like creates array with matching shape and dtype."""
        host = np.arange(20, dtype=np.int32).reshape(4, 5)
        dev = cuda.device_array_like(host)
        self.assertEqual(dev.shape, host.shape)
        self.assertEqual(dev.dtype, host.dtype)

    def test_to_device_copy_to_host_roundtrip(self):
        """Verify data integrity in to_device -> copy_to_host roundtrip."""
        host_orig = np.arange(100, dtype=np.float32)
        dev = cuda.to_device(host_orig)
        host_copy = dev.copy_to_host()
        self.assertTrue(np.array_equal(host_orig, host_copy))

    def test_copy_to_device_existing_array(self):
        """Verify copy_to_device into pre-allocated device array."""
        host = np.arange(50, dtype=np.int32)
        dev = cuda.device_array(50, dtype=np.int32)
        dev.copy_to_device(host)
        result = dev.copy_to_host()
        self.assertTrue(np.array_equal(host, result))

    @skip_on_cudasim("Device-to-device copy requires real device memory")
    def test_device_to_device_copy(self):
        """Verify device-to-device memory copy."""
        n = 32
        host = np.arange(n, dtype=np.float32)
        dev1 = cuda.to_device(host)
        dev2 = cuda.device_array(n, dtype=np.float32)
        
        # Copy from dev1 to dev2
        dev2.copy_to_device(dev1)
        result = dev2.copy_to_host()
        self.assertTrue(np.array_equal(host, result))

    def test_multidimensional_shape_consistency(self):
        """Verify shape consistency for multidimensional arrays."""
        shapes = [(10,), (5, 8), (3, 4, 5), (2, 3, 4, 5)]
        for shape in shapes:
            arr = cuda.device_array(shape, dtype=np.float32)
            self.assertEqual(arr.shape, shape)
            self.assertEqual(arr.ndim, len(shape))
            self.assertEqual(arr.size, np.prod(shape))

    @pytest.mark.xfail(reason="reshape not supported on CUDA device arrays")
    def test_reshape_device_array(self):
        """reshape is not currently supported on DeviceArray."""
        arr = cuda.device_array((4, 8), dtype=np.int32)
        _ = arr.reshape((32,))

    @pytest.mark.xfail(reason="view not supported on CUDA device arrays")
    def test_view_device_array(self):
        """view is not currently supported on DeviceArray."""
        arr = cuda.device_array(16, dtype=np.int32)
        _ = arr.view(np.float32)

    @pytest.mark.xfail(reason="ravel not supported on CUDA device arrays")
    def test_ravel_device_array(self):
        """ravel is not currently supported on DeviceArray."""
        arr = cuda.device_array((4, 8), dtype=np.int32)
        _ = arr.ravel()
