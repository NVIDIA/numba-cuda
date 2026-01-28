# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from collections import namedtuple
from functools import partial
from itertools import product
from numba.cuda import vectorize as cuda_vectorize
from numba import cuda, vectorize as numba_vectorize
from numba.cuda.types import int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
import cupy as cp
import pytest
from numba.cuda.testing import DeprecatedDeviceArrayApiWarning


# Signatures to test with - these are all homogeneous in dtype, so the output
# dtype should match the input dtype - the output should not have been cast
# upwards, as reported in #8400: https://github.com/numba/numba/issues/8400
signatures = [
    int32(int32, int32),
    float32(float32, float32),
    float64(float64, float64),
]

# The order here is chosen such that each subsequent dtype might have been
# casted to a previously-used dtype. This is unlikely to be an issue for CUDA,
# but there might be future circumstances in which it becomes relevant, perhaps
# if it supported Dynamic UFuncs, and we want to ensure that an implementation
# for a the given dtype is used rather than casting the input upwards.
dtypes = (np.float64, np.float32, np.int32)

# NumPy ndarray orders
orders = ("C", "F")

# Input sizes corresponding to operations:
# - Less than one warp,
# - Less than one block,
# - Greater than one block (i.e. many blocks)
input_sizes = (8, 100, 2**10 + 1)

# Vectorize functions to test
# cuda.vectorize doesn't need target parameter, numba.vectorize needs target="cuda"
vectorize_funcs = [cuda_vectorize, partial(numba_vectorize, target="cuda")]


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestCUDAVectorize(CUDATestCase):
    # Presumably chosen as an odd number unlikely to coincide with the total
    # thread count, and large enough to ensure a significant number of blocks
    # are used.
    N = 1000001

    def test_scalar(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            a = 1.2
            b = 2.3
            c = vector_add(a, b)
            self.assertEqual(c, a + b)

    def test_1d(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            for ty in dtypes:
                data = np.array(np.random.random(self.N), dtype=ty)
                expected = np.add(data, data)
                actual = vector_add(data, data)
                np.testing.assert_allclose(expected, actual)
                self.assertEqual(actual.dtype, ty)

    def test_1d_async(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            nb_stream = cuda.stream()
            stream = cp.cuda.Stream()

            for ty in dtypes:
                data = np.array(np.random.random(self.N), dtype=ty)
                with stream:
                    device_data = cp.asarray(data)

                dresult = vector_add(device_data, device_data, stream=nb_stream)
                actual = dresult.copy_to_host()

                expected = np.add(data, data)

                np.testing.assert_allclose(expected, actual)
                self.assertEqual(actual.dtype, ty)

    def test_nd(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            for nd, dtype, order in product(range(1, 8), dtypes, orders):
                shape = (4,) * nd
                data = np.random.random(shape).astype(dtype)
                data2 = np.array(data.T, order=order)

                expected = data + data2
                actual = vector_add(data, data2)
                np.testing.assert_allclose(expected, actual)
                self.assertEqual(actual.dtype, dtype)

    def test_output_arg(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            A = np.arange(10, dtype=np.float32)
            B = np.arange(10, dtype=np.float32)

            expected = A + B
            actual = np.empty_like(A)
            vector_add(A, B, out=actual)

            np.testing.assert_allclose(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)

    def test_reduce(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            dtype = np.int32

            for n in input_sizes:
                x = np.arange(n, dtype=dtype)
                expected = np.add.reduce(x)
                actual = vector_add.reduce(x)
                np.testing.assert_allclose(expected, actual)
                # np.add.reduce is special-cased to return an int64 for any int
                # arguments, so we can't compare against its returned dtype when
                # we're checking the general reduce machinery (which just happens
                # to be using addition). Instead, compare against the input dtype.
                self.assertEqual(dtype, actual.dtype)

    def test_reduce_async(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            nb_stream = cuda.stream()
            stream = cp.cuda.Stream()
            dtype = np.int32

            for n in input_sizes:
                x = np.arange(n, dtype=dtype)
                expected = np.add.reduce(x)
                with stream:
                    dx = cp.asarray(x)
                actual = vector_add.reduce(dx, stream=nb_stream)
                np.testing.assert_allclose(expected, actual)
                # Compare against the input dtype as in test_reduce().
                self.assertEqual(dtype, actual.dtype)

    def test_manual_transfer(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            n = 10
            x = np.arange(n, dtype=np.int32)
            dx = cp.asarray(x)
            expected = x + x
            actual = vector_add(x, dx).copy_to_host()
            np.testing.assert_equal(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)

    def test_ufunc_output_2d(self):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            n = 10
            x = np.arange(n, dtype=np.int32).reshape(2, 5)
            dx = cp.asarray(x)
            vector_add(dx, dx, out=dx)

            expected = x + x
            actual = dx.get()
            np.testing.assert_equal(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)

    def check_tuple_arg(self, a, b):
        for vectorize in vectorize_funcs:

            @vectorize(signatures)
            def vector_add(a, b):
                return a + b

            r = vector_add(a, b)
            np.testing.assert_equal(np.asarray(a) + np.asarray(b), r)

    def test_tuple_arg(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        self.check_tuple_arg(a, b)

    def test_namedtuple_arg(self):
        Point = namedtuple("Point", ("x", "y", "z"))
        a = Point(x=1.0, y=2.0, z=3.0)
        b = Point(x=4.0, y=5.0, z=6.0)
        self.check_tuple_arg(a, b)

    def test_tuple_of_array_arg(self):
        arr = np.arange(10, dtype=np.int32)
        a = (arr, arr + 1)
        b = (arr + 2, arr + 2)
        self.check_tuple_arg(a, b)

    def test_tuple_of_namedtuple_arg(self):
        Point = namedtuple("Point", ("x", "y", "z"))
        a = (Point(x=1.0, y=2.0, z=3.0), Point(x=1.5, y=2.5, z=3.5))
        b = (Point(x=4.0, y=5.0, z=6.0), Point(x=4.5, y=5.5, z=6.5))
        self.check_tuple_arg(a, b)

    def test_namedtuple_of_array_arg(self):
        xs1 = np.arange(10, dtype=np.int32)
        ys1 = xs1 + 2
        xs2 = np.arange(10, dtype=np.int32) * 2
        ys2 = xs2 + 1
        Points = namedtuple("Points", ("xs", "ys"))
        a = Points(xs=xs1, ys=ys1)
        b = Points(xs=xs2, ys=ys2)
        self.check_tuple_arg(a, b)

    def test_name_attribute(self):
        for vectorize in vectorize_funcs:

            @vectorize("f8(f8)")
            def bar(x):
                return x**2

            self.assertEqual(bar.__name__, "bar")


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestCUDAVectorizeNoTransfer(CUDATestCase):
    """Test that vectorize operations on device data don't induce transfers."""

    def setUp(self):
        """Set up mocks to block host <-> device transfers."""
        super().setUp()

        # Initialize test data on the device prior to banning host <-> device
        # transfer
        self.noise = np.random.randn(1, 3, 64, 64).astype(np.float32)
        with pytest.warns(DeprecatedDeviceArrayApiWarning):
            self.device_noise = cuda.to_device(self.noise)

        # A mock of a CUDA function that always raises a CudaAPIError
        def raising_transfer(*args, **kwargs):
            raise CudaAPIError(999, "Transfer not allowed")

        # Save the original implementations
        self.old_HtoD = getattr(driver, "cuMemcpyHtoD", None)
        self.old_DtoH = getattr(driver, "cuMemcpyDtoH", None)

        # Replace with mocks that prevent transfers
        driver.cuMemcpyHtoD = raising_transfer
        driver.cuMemcpyDtoH = raising_transfer

    def tearDown(self):
        """Restore original transfer functions."""
        # Replace our mocks with the original implementations. If there was
        # no original implementation, simply remove ours.
        if self.old_HtoD is not None:
            driver.cuMemcpyHtoD = self.old_HtoD
        else:
            if hasattr(driver, "cuMemcpyHtoD"):
                del driver.cuMemcpyHtoD

        if self.old_DtoH is not None:
            driver.cuMemcpyDtoH = self.old_DtoH
        else:
            if hasattr(driver, "cuMemcpyDtoH"):
                del driver.cuMemcpyDtoH

        super().tearDown()

    def test_mock_blocks_device_to_host_transfer(self):
        """Verify that the mock successfully blocks device-to-host transfers."""
        with self.assertRaisesRegex(CudaAPIError, "Transfer not allowed"):
            self.device_noise.copy_to_host()

    def test_mock_blocks_host_to_device_transfer(self):
        """Verify that the mock successfully blocks host-to-device transfers."""
        with self.assertRaisesRegex(CudaAPIError, "Transfer not allowed"):
            with pytest.warns(DeprecatedDeviceArrayApiWarning):
                cuda.to_device([1])

    def test_vectorize_with_device_data_no_transfer(self):
        """Test that vectorize operations on device data don't induce transfers."""
        for vectorize in vectorize_funcs:
            # Define and call a ufunc with data on the device
            # This should not induce any transfers (which would raise CudaAPIError)
            @vectorize(["float32(float32)"])
            def func(noise):
                return noise + 1.0

            # This should succeed without raising CudaAPIError
            result = func(self.device_noise)

            # Verify the result is still on the device and has the right shape
            self.assertEqual(result.shape, self.device_noise.shape)


if __name__ == "__main__":
    unittest.main()
