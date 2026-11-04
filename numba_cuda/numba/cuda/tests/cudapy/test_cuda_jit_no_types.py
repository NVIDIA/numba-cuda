# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.support import override_config
import unittest


class TestCudaJitNoTypes(CUDATestCase):
    """
    Tests the jit decorator with no types provided.
    """

    def test_device_array(self):
        @cuda.jit
        def foo(x, y):
            i = cuda.grid(1)
            y[i] = x[i]

        x = np.arange(10)
        y = np.empty_like(x)

        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        foo[10, 1](dx, dy)

        dy.copy_to_host(y)

        self.assertTrue(np.all(x == y))

    def test_device_jit(self):
        @cuda.jit(device=True)
        def mapper(args):
            a, b, c = args
            return a + b + c

        @cuda.jit(device=True)
        def reducer(a, b):
            return a + b

        @cuda.jit
        def driver(A, B):
            i = cuda.grid(1)
            if i < B.size:
                args = A[i], A[i] + B[i], B[i]
                B[i] = reducer(mapper(args), 1)

        A = np.arange(100, dtype=np.float32)
        B = np.arange(100, dtype=np.float32)

        Acopy = A.copy()
        Bcopy = B.copy()

        driver[1, 100](A, B)

        np.testing.assert_allclose(Acopy + Acopy + Bcopy + Bcopy + 1, B)

    def test_device_jit_2(self):
        @cuda.jit(device=True)
        def inner(arg):
            return arg + 1

        @cuda.jit
        def outer(argin, argout):
            argout[0] = inner(argin[0]) + inner(2)

        a = np.zeros(1)
        b = np.zeros(1)

        stream = cuda.stream()
        d_a = cuda.to_device(a, stream)
        d_b = cuda.to_device(b, stream)

        outer[1, 1, stream](d_a, d_b)

        d_b.copy_to_host(b, stream)

        self.assertEqual(b[0], (a[0] + 1) + (2 + 1))

    def test_jit_debug_simulator(self):
        # Ensure that the jit decorator accepts the debug kwarg when the
        # simulator is in use - see Issue #6615.
        with override_config("ENABLE_CUDASIM", 1):

            @cuda.jit(debug=True, opt=False)
            def f(x):
                pass

    def test_jit_lto_config_default_and_override(self):
        # Issue #162: Provide an option to control default LTO mode via
        # config.CUDA_ENABLE_LTO
        from unittest.mock import patch

        def dummy_kernel():
            pass

        # Verify default configuration value is enabled (1)
        self.assertEqual(getattr(config, "CUDA_ENABLE_LTO", 1), 1)

        # 1. When nvjitlink is available:
        with patch("numba.cuda.decorators._have_nvjitlink", return_value=True):
            # Default with CUDA_ENABLE_LTO=1 should default lto to True
            with override_config("CUDA_ENABLE_LTO", 1):
                disp = cuda.jit(dummy_kernel)
                self.assertTrue(disp.targetoptions["lto"])

            # Disabling CUDA_ENABLE_LTO should default lto to False
            with override_config("CUDA_ENABLE_LTO", 0):
                disp = cuda.jit(dummy_kernel)
                self.assertFalse(disp.targetoptions["lto"])

                # Explicit lto=True should still be respected even when
                # CUDA_ENABLE_LTO=0
                disp_explicit_true = cuda.jit(dummy_kernel, lto=True)
                self.assertTrue(disp_explicit_true.targetoptions["lto"])

            # Explicit lto=False should still be respected even when
            # CUDA_ENABLE_LTO=1
            with override_config("CUDA_ENABLE_LTO", 1):
                disp_explicit_false = cuda.jit(dummy_kernel, lto=False)
                self.assertFalse(disp_explicit_false.targetoptions["lto"])

            # When debug=True, lto is disabled regardless of CUDA_ENABLE_LTO
            with override_config("CUDA_ENABLE_LTO", 1):
                disp_debug = cuda.jit(dummy_kernel, debug=True, opt=False)
                self.assertFalse(disp_debug.targetoptions["lto"])

        # 2. When nvjitlink is NOT available:
        with patch("numba.cuda.decorators._have_nvjitlink", return_value=False):
            # Even if CUDA_ENABLE_LTO=1, lto defaults to False
            with override_config("CUDA_ENABLE_LTO", 1):
                disp = cuda.jit(dummy_kernel)
                self.assertFalse(disp.targetoptions["lto"])

            # Explicit lto=True raises RuntimeError
            with self.assertRaisesRegex(RuntimeError, "LTO requires nvjitlink"):
                cuda.jit(dummy_kernel, lto=True)


if __name__ == "__main__":
    unittest.main()
