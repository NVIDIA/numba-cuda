# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Test cases adapted from numba/tests/test_enums.py
"""

import numpy as np

from numba import int16, int32
from numba import cuda, vectorize, njit
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba_cuda_tests.enum_usecases import (
    Color,
    Shape,
    Planet,
    RequestError,
    IntEnumWithNegatives,
)


class EnumTest(CUDATestCase):
    pairs = [
        (Color.red, Color.red),
        (Color.red, Color.green),
        (Planet.EARTH, Planet.EARTH),
        (Planet.VENUS, Planet.MARS),
        (Shape.circle, IntEnumWithNegatives.two),  # IntEnum, same value
    ]

    def test_compare(self):
        def f(a, b, out):
            out[0] = a == b
            out[1] = a != b
            out[2] = a is b
            out[3] = a is not b

        cuda_f = cuda.jit(f)
        for a, b in self.pairs:
            got = np.zeros((4,), dtype=np.bool_)
            expected = got.copy()
            cuda_f[1, 1](a, b, got)
            f(a, b, expected)
            self.assertPreciseEqual(expected, got)

    def test_getattr_getitem(self):
        def f(out):
            # Lookup of an enum member on its class
            out[0] = Color.red == Color.green
            out[1] = Color["red"] == Color["green"]

        cuda_f = cuda.jit(f)
        got = np.zeros((2,), dtype=np.bool_)
        expected = got.copy()
        cuda_f[1, 1](got)
        f(expected)
        self.assertPreciseEqual(expected, got)

    def test_return_from_device_func(self):
        @njit
        def inner(pred):
            return Color.red if pred else Color.green

        def f(pred, out):
            out[0] = inner(pred) == Color.red
            out[1] = inner(not pred) == Color.green

        cuda_f = cuda.jit(f)
        got = np.zeros((2,), dtype=np.bool_)
        expected = got.copy()
        f(True, expected)
        cuda_f[1, 1](True, got)
        self.assertPreciseEqual(expected, got)

    def test_int_coerce(self):
        def f(x, out):
            # Implicit coercion of intenums to ints
            if x > RequestError.internal_error:
                out[0] = x - RequestError.not_found
            else:
                out[0] = x + Shape.circle

        cuda_f = cuda.jit(f)
        for x in [300, 450, 550]:
            got = np.zeros((1,), dtype=np.int32)
            expected = got.copy()
            cuda_f[1, 1](x, got)
            f(x, expected)
            self.assertPreciseEqual(expected, got)

    def test_int_cast(self):
        def f(x, out):
            # Explicit coercion of intenums to ints
            if x > int16(RequestError.internal_error):
                out[0] = x - int32(RequestError.not_found)
            else:
                out[0] = x + int16(Shape.circle)

        cuda_f = cuda.jit(f)
        for x in [300, 450, 550]:
            got = np.zeros((1,), dtype=np.int32)
            expected = got.copy()
            cuda_f[1, 1](x, got)
            f(x, expected)
            self.assertEqual(expected, got)

    @skip_on_cudasim("ufuncs are unsupported on simulator.")
    def test_vectorize(self):
        def f(x):
            if x != RequestError.not_found:
                return RequestError["internal_error"]
            else:
                return RequestError.dummy

        cuda_func = vectorize("int64(int64)", target="cuda")(f)
        arr = np.array([2, 404, 500, 404], dtype=np.int64)
        expected = np.array([f(x) for x in arr], dtype=np.int64)
        got = cuda_func(arr)
        self.assertPreciseEqual(expected, got)

    @skip_on_cudasim("No typing context in CUDA simulator")
    def test_int_enum_no_conversion(self):
        # Ported from Numba PR #10047: "Fix IntEnumMember.can_convert_to() when
        # no conversions found", https://github.com/numba/numba/pull/10047.

        # The original test is intended to ensures that
        # IntEnumMember.can_convert_to() handles the case when the typing
        # context's can_convert() method returns None to signal no possible
        # conversion. In Numba-CUDA, we had to patch the CUDA target context to
        # work around this issue, because we cannot guarantee that the
        # IntEnumMember method can be patched before instances are created.
        ctx = cuda.descriptor.cuda_target.typing_context

        int_enum_type = types.IntEnumMember(Shape, types.int64)
        # Conversion of an int enum member to a 1D array would be invalid
        invalid_toty = types.int64[::1]
        self.assertIsNone(ctx.can_convert(int_enum_type, invalid_toty))


if __name__ == "__main__":
    unittest.main()
