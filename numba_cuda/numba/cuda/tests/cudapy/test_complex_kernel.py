# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import HAS_NUMBA, types
from numba.cuda.core.config import ENABLE_CUDASIM

if HAS_NUMBA:
    from numba.core.errors import TypingError
else:
    from numba.cuda.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestNumpyNumberConstructorCompile(unittest.TestCase):
    def test_complex_two_arguments(self):
        component_types = (
            types.float32,
            types.float64,
            types.int32,
            types.int64,
            types.uint64,
        )
        for dtype, expected in (
            (np.complex64, types.complex64),
            (np.complex128, types.complex128),
        ):

            def constructor(real, imag):
                return dtype(real, imag)

            for real in component_types:
                for imag in component_types:
                    with self.subTest(dtype=dtype, real=real, imag=imag):
                        _, resty = cuda.compile_ptx(
                            constructor, (real, imag), device=True, cc=(7, 5)
                        )
                        self.assertEqual(resty, expected)

    def test_one_argument(self):
        for dtype, expected in (
            (np.int32, types.int32),
            (np.int64, types.int64),
            (np.float32, types.float32),
            (np.float64, types.float64),
            (np.complex64, types.complex64),
            (np.complex128, types.complex128),
        ):

            def constructor(value):
                return dtype(value)

            for arg in (types.int32, types.float64, types.boolean):
                with self.subTest(dtype=dtype, arg=arg):
                    _, resty = cuda.compile_ptx(
                        constructor, (arg,), device=True, cc=(7, 5)
                    )
                    self.assertEqual(resty, expected)

    def test_invalid_arguments(self):
        for dtype in (np.complex64, np.complex128):

            def no_arguments():
                return dtype()

            def three_arguments(real, imag, extra):
                return dtype(real, imag, extra)

            def two_arguments(real, imag):
                return dtype(real, imag)

            cases = [
                (no_arguments, ()),
                (three_arguments, (types.float64,) * 3),
                (two_arguments, (types.complex64, types.float64)),
                (two_arguments, (types.complex128, types.float64)),
                (two_arguments, (types.float64, types.complex64)),
                (two_arguments, (types.float64, types.complex128)),
                (two_arguments, (types.boolean, types.float64)),
                (two_arguments, (types.float64, types.none)),
            ]
            for constructor, sig in cases:
                with self.subTest(dtype=dtype, sig=sig):
                    with self.assertRaises(TypingError):
                        cuda.compile_ptx(
                            constructor, sig, device=True, cc=(7, 5)
                        )

        for dtype in (np.int32, np.float64):

            def constructor(real, imag):
                return dtype(real, imag)

            with self.subTest(dtype=dtype):
                with self.assertRaises(TypingError):
                    cuda.compile_ptx(
                        constructor,
                        (types.float64,) * 2,
                        device=True,
                        cc=(7, 5),
                    )


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
        # The simulator executes NumPy directly instead of type checking.
        error = TypeError if ENABLE_CUDASIM else TypingError
        for dtype in (np.complex64, np.complex128):

            @cuda.jit
            def foo(out, real, imag):
                out[0] = dtype(real, imag)

            out = np.empty(1, dtype=dtype)
            for real, imag in ((1j, 2.0), (1.0, 2j)):
                with self.assertRaises(error):
                    foo[1, 1](out, real, imag)


if __name__ == "__main__":
    unittest.main()
