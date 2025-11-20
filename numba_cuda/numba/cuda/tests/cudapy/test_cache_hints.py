# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda, errors, typeof, types
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np

tested_types = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)

complex_types = (np.complex64, np.complex128)

load_operators = (
    (cuda.ldca, "ca"),
    (cuda.ldcg, "cg"),
    (cuda.ldcs, "cs"),
    (cuda.ldlu, "lu"),
    (cuda.ldcv, "cv"),
)
store_operators = (
    (cuda.stcg, "cg"),
    (cuda.stcs, "cs"),
    (cuda.stwb, "wb"),
    (cuda.stwt, "wt"),
)


class TestCacheHints(CUDATestCase):
    def test_loads(self):
        for operator, modifier in load_operators:

            @cuda.jit
            def f(r, x):
                for i in range(len(r)):
                    r[i] = operator(x, i)

            def f_ptr(ptr, r, n):
                for i in range(n):
                    r[i] = operator(ptr, i)

            for ty in tested_types:
                x = np.arange(5).astype(ty)
                numba_type = typeof(x)
                bitwidth = numba_type.dtype.bitwidth

                with self.subTest(operator=operator, ty=ty, input_type="array"):
                    r = np.zeros_like(x)
                    f[1, 1](r, x)
                    np.testing.assert_equal(r, x)

                    # Check PTX contains a cache-policy load instruction
                    sig = (numba_type, numba_type)
                    ptx = f.inspect_asm(signature=sig)

                    self.assertIn(f"ld.global.{modifier}.b{bitwidth}", ptx)

                with self.subTest(
                    operator=operator, ty=ty, input_type="cpointer"
                ):
                    ptr_type = types.CPointer(numba_type.dtype)
                    
                    sig = (ptr_type, numba_type, types.intp)
                    ptx, _ = cuda.compile_ptx(f_ptr, sig)

                    self.assertIn(f"ld.global.{modifier}.b{bitwidth}", ptx)

    def test_stores(self):
        for operator, modifier in store_operators:

            @cuda.jit
            def f(r, x):
                for i in range(len(r)):
                    operator(r, i, x[i])

            def f_ptr(ptr, values, n):
                for i in range(n):
                    operator(ptr, i, values[i])

            for ty in tested_types:
                x = np.arange(5).astype(ty)
                numba_type = typeof(x)
                bitwidth = numba_type.dtype.bitwidth

                with self.subTest(operator=operator, ty=ty):
                    r = np.zeros_like(x)
                    f[1, 1](r, x)
                    np.testing.assert_equal(r, x)

                    # Check PTX contains a cache-policy store instruction
                    sig = (numba_type, numba_type)
                    ptx = f.inspect_asm(signature=sig)

                    self.assertIn(f"st.global.{modifier}.b{bitwidth}", ptx)

                with self.subTest(
                    operator=operator, ty=ty, input_type="cpointer"
                ):
                    ptr_type = types.CPointer(numba_type.dtype)

                    sig = (ptr_type, numba_type, types.intp)
                    ptx, _ = cuda.compile_ptx(f_ptr, sig)

                    self.assertIn(f"st.global.{modifier}.b{bitwidth}", ptx)

    def test_bad_indices(self):
        def float_indices(x):
            cuda.ldcs(x, 1.0)

        sig_1d = (types.float32[::1],)

        msg = "float64 is not a valid index"
        with self.assertRaisesRegex(errors.TypingError, msg):
            cuda.compile_ptx(float_indices, sig_1d)

        def too_long_indices(x):
            cuda.ldcs(x, (1, 2))

        msg = "Expected 1 indices, got 2"
        with self.assertRaisesRegex(errors.TypingError, msg):
            cuda.compile_ptx(too_long_indices, sig_1d)

        def too_short_indices_scalar(x):
            cuda.ldcs(x, 1)

        def too_short_indices_tuple(x):
            cuda.ldcs(x, (1,))

        sig_2d = (types.float32[:, ::1],)

        msg = "Expected 2 indices, got a scalar"
        with self.assertRaisesRegex(errors.TypingError, msg):
            cuda.compile_ptx(too_short_indices_scalar, sig_2d)

        msg = "Expected 2 indices, got 1"
        with self.assertRaisesRegex(errors.TypingError, msg):
            cuda.compile_ptx(too_short_indices_tuple, sig_2d)

    def test_unsupported_dtypes(self):
        def load_complex(r, x):
            r[0] = cuda.ldcs(x, 0)

        # complex types are not supported (not integer or float)
        numba_type = typeof(np.zeros(1, dtype=np.complex64))
        sig = (numba_type, numba_type)

        msg = "ldcs requires array of integer or float type"
        with self.assertRaisesRegex(errors.TypingError, msg):
            cuda.compile_ptx(load_complex, sig)


if __name__ == "__main__":
    unittest.main()
