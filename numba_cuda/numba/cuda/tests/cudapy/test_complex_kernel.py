# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba.core.errors import TypingError
else:
    from numba.cuda.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase


class TestCudaComplex(CUDATestCase):
    def test_cuda_complex_arg(self):
        @cuda.jit("void(complex128[:], complex128)")
        def foo(a, b):
            i = cuda.grid(1)
            a[i] += b

        a = np.arange(5, dtype=np.complex128)
        a0 = a.copy()
        foo[1, a.shape](a, 2j)
        self.assertTrue(np.allclose(a, a0 + 2j))

    def test_numpy_complex_constructor_two_args(self):
        for dtype, component_dtype in (
            (np.complex64, np.float32),
            (np.complex128, np.float64),
        ):

            @cuda.jit
            def foo(out, real, imag):
                i = cuda.grid(1)
                if i < out.size:
                    out[i] = dtype(real[i], imag[i])

            real = np.array([1.5, -2.0, 0.0], dtype=component_dtype)
            imag = np.array([-3.25, 4.0, 0.5], dtype=component_dtype)
            out = np.empty(real.size, dtype=dtype)
            foo[1, real.size](out, real, imag)

            self.assertTrue(np.allclose(out, real + 1j * imag))

    def test_numpy_complex_constructor_rejects_complex_components(self):
        for dtype in (np.complex64, np.complex128):

            @cuda.jit
            def foo(out, real, imag):
                out[0] = dtype(real, imag)

            out = np.empty(1, dtype=dtype)
            for real, imag in ((1j, 2.0), (1.0, 2j)):
                with self.assertRaises(TypingError):
                    foo[1, 1](out, real, imag)


if __name__ == "__main__":
    unittest.main()
