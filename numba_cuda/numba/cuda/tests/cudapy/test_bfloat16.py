# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
import numpy as np
from ml_dtypes import bfloat16 as mldtypes_bf16
from numba import cuda
from numba.cuda import (
    float32,
    float64,
    int16,
    int32,
    int64,
    uint16,
    uint32,
    uint64,
)
from numba.cuda import config

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp

if not config.ENABLE_CUDASIM:
    from numba.cuda.bf16 import (
        bfloat16,
        habs,
        hadd,
        hsub,
        hmul,
        hadd_rn,
        hsub_rn,
        hmul_rn,
        hdiv,
        hadd_sat,
        hsub_sat,
        hmul_sat,
        hfma,
        hfma_sat,
        hneg,
        hfma_relu,
        # Comparison intrinsics
        heq,
        hne,
        hge,
        hgt,
        hle,
        hlt,
        hmax,
        hmin,
        hmax_nan,
        hmin_nan,
        hisnan,
        hisinf,
        # Conversion intrinsics (NumPy-style names)
        bfloat16_to_int8_rz,
        bfloat16_to_uint8_rz,
        int16_to_bfloat16_rn,
        int16_to_bfloat16_rz,
        int16_to_bfloat16_rd,
        int16_to_bfloat16_ru,
        bfloat16_to_int16_rn,
        bfloat16_to_int16_rz,
        bfloat16_to_int16_rd,
        bfloat16_to_int16_ru,
        uint16_to_bfloat16_rn,
        uint16_to_bfloat16_rz,
        uint16_to_bfloat16_rd,
        uint16_to_bfloat16_ru,
        bfloat16_to_uint16_rn,
        bfloat16_to_uint16_rz,
        bfloat16_to_uint16_rd,
        bfloat16_to_uint16_ru,
        int32_to_bfloat16_rn,
        int32_to_bfloat16_rz,
        int32_to_bfloat16_rd,
        int32_to_bfloat16_ru,
        bfloat16_to_int32_rn,
        bfloat16_to_int32_rz,
        bfloat16_to_int32_rd,
        bfloat16_to_int32_ru,
        uint32_to_bfloat16_rn,
        uint32_to_bfloat16_rz,
        uint32_to_bfloat16_rd,
        uint32_to_bfloat16_ru,
        bfloat16_to_uint32_rn,
        bfloat16_to_uint32_rz,
        bfloat16_to_uint32_rd,
        bfloat16_to_uint32_ru,
        bfloat16_to_int64_rn,
        bfloat16_to_int64_rz,
        bfloat16_to_int64_rd,
        bfloat16_to_int64_ru,
        int64_to_bfloat16_rn,
        int64_to_bfloat16_rz,
        int64_to_bfloat16_rd,
        int64_to_bfloat16_ru,
        bfloat16_to_uint64_rn,
        bfloat16_to_uint64_rz,
        bfloat16_to_uint64_rd,
        bfloat16_to_uint64_ru,
        uint64_to_bfloat16_rn,
        uint64_to_bfloat16_rz,
        uint64_to_bfloat16_rd,
        uint64_to_bfloat16_ru,
        bfloat16_as_int16,
        int16_as_bfloat16,
        bfloat16_as_uint16,
        uint16_as_bfloat16,
        bfloat16_to_float32,
        float32_to_bfloat16,
        float64_to_bfloat16,
        float32_to_bfloat16_rn,
        float32_to_bfloat16_rz,
        float32_to_bfloat16_rd,
        float32_to_bfloat16_ru,
    )

from numba.cuda.testing import CUDATestCase

import math


class TestBfloat16HighLevelBindings(CUDATestCase):
    def skip_unsupported(self):
        if not cuda.is_bfloat16_supported():
            self.skipTest("bfloat16 requires compute capability 8.0+")

    def test_use_type_in_kernel(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel():
            bfloat16(3.14)

        kernel[1, 1]()

    def test_math_bindings(self):
        self.skip_unsupported()

        exp_functions = [math.exp]
        if sys.version_info >= (3, 11):
            exp_functions += [math.exp2]

        functions = [
            math.trunc,
            math.ceil,
            math.floor,
            math.sqrt,
            math.log,
            math.log10,
            math.cos,
            math.sin,
            math.tanh,
        ] + exp_functions

        for f in functions:
            with self.subTest(func=f):

                @cuda.jit
                def kernel(arr):
                    x = bfloat16(3.14)
                    y = f(x)
                    arr[0] = float32(y)

                arr = cp.zeros((1,), dtype="float32")
                kernel[1, 1](arr)

                if f in exp_functions:
                    self.assertAlmostEqual(arr[0], f(3.14), delta=1e-1)
                else:
                    self.assertAlmostEqual(arr[0], f(3.14), delta=1e-2)

    def test_arithmetic_intrinsics_basic(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(1.25)
            b = bfloat16(-2.5)

            out[0] = float32(habs(b))
            out[1] = float32(hadd(a, b))
            out[2] = float32(hsub(a, b))
            out[3] = float32(hmul(a, b))
            out[4] = float32(hdiv(b, a))
            out[5] = float32(hneg(a))
            out[6] = float32(hfma(a, b, b))

            out[7] = float32(hadd_rn(a, b))
            out[8] = float32(hsub_rn(a, b))
            out[9] = float32(hmul_rn(a, b))

        out = cp.zeros((10,), dtype="float32")
        kernel[1, 1](out)

        a = 1.25
        b = -2.5
        expected = [
            abs(b),
            a + b,
            a - b,
            a * b,
            b / a,
            -a,
            a * b + b,
            a + b,
            a - b,
            a * b,
        ]
        for i, exp in enumerate(expected):
            self.assertAlmostEqual(out[i], exp, delta=1e-2)

    def test_arithmetic_intrinsics_saturating(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(1.5)
            b = bfloat16(0.75)

            out[0] = float32(hadd_sat(a, b))  # 2.25 -> 1.0
            out[1] = float32(hsub_sat(b, a))  # -0.75 -> 0.0
            out[2] = float32(hmul_sat(a, b))  # 1.125 -> 1.0
            out[3] = float32(hfma_sat(a, b, a))  # 1.125 + 1.5 -> 1.0

        out = cp.zeros((4,), dtype="float32")
        kernel[1, 1](out)

        self.assertAlmostEqual(out[0], 1.0, delta=1e-3)
        self.assertAlmostEqual(out[1], 0.0, delta=1e-3)
        self.assertAlmostEqual(out[2], 1.0, delta=1e-3)
        self.assertAlmostEqual(out[3], 1.0, delta=1e-3)

        # Also check they are clamped within [0, 1]
        for i in range(4):
            self.assertGreaterEqual(out[i], 0.0)
            self.assertLessEqual(out[i], 1.0)

    def test_fma_relu_intrinsic(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(-1.5)
            b = bfloat16(2.0)
            c = bfloat16(0.0)

            out[0] = float32(hfma_relu(a, b, c))  # -3.0 -> relu -> 0.0

        out = cp.zeros((1,), dtype="float32")
        kernel[1, 1](out)

        self.assertAlmostEqual(out[0], 0.0, delta=1e-3)

    def test_comparison_intrinsics(self):
        self.skip_unsupported()

        def make_kernel(cmpfn):
            @cuda.jit
            def kernel(out, a, b):
                a_bf16 = bfloat16(a)
                b_bf16 = bfloat16(b)
                out[0] = cmpfn(a_bf16, b_bf16)

            return kernel

        comparisons = [heq, hne, hge, hgt, hle, hlt]
        ops = [
            lambda x, y: x == y,
            lambda x, y: x != y,
            lambda x, y: x >= y,
            lambda x, y: x > y,
            lambda x, y: x <= y,
            lambda x, y: x < y,
        ]

        for cmpfn, op in zip(comparisons, ops):
            with self.subTest(cmpfn=cmpfn):
                kernel = make_kernel(cmpfn)
                out = cp.zeros((1,), dtype="bool")

                a = 3.0
                b = 3.0
                kernel[1, 1](out, a, b)
                self.assertEqual(bool(out[0]), op(3.0, 3.0))

                a = 3.0
                b = 4.0
                kernel[1, 1](out, a, b)
                self.assertEqual(bool(out[0]), op(3.0, 4.0))

                a = 4.0
                b = 3.0
                kernel[1, 1](out, a, b)
                self.assertEqual(bool(out[0]), op(4.0, 3.0))

    def test_hmax_hmin_intrinsics(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(3.0)
            b = bfloat16(4.0)
            out[0] = float32(hmax(a, b))
            out[1] = float32(hmin(a, b))

        out = cp.zeros((2,), dtype="float32")
        kernel[1, 1](out)
        self.assertAlmostEqual(out[0], 4.0, delta=1e-3)
        self.assertAlmostEqual(out[1], 3.0, delta=1e-3)

    def test_nan_and_inf_intrinsics(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out_bool, out_int):
            nanv = bfloat16(float("nan"))
            infv = bfloat16(float("inf"))
            out_bool[0] = hisnan(nanv)
            out_int[0] = hisinf(infv)

        out_bool = cp.zeros((1,), dtype="bool")
        out_int = cp.zeros((1,), dtype="int32")
        kernel[1, 1](out_bool, out_int)
        self.assertTrue(bool(out_bool[0]))
        self.assertNotEqual(int(out_int[0]), 0)

    def test_hmax_nan_hmin_nan_intrinsics(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(float("nan"))
            b = bfloat16(2.0)
            out[0] = float32(hmax_nan(a, b))
            out[1] = float32(hmin_nan(a, b))
            out[2] = float32(hmax(a, b))
            out[3] = float32(hmin(a, b))

        out = cp.zeros((4,), dtype="float32")
        kernel[1, 1](out)
        # NaN-propagating variants should produce NaN
        self.assertTrue(math.isnan(out[0]))
        self.assertTrue(math.isnan(out[1]))
        # Non-NaN variants should return the non-NaN operand
        self.assertAlmostEqual(out[2], 2.0, delta=1e-3)
        self.assertAlmostEqual(out[3], 2.0, delta=1e-3)

    def test_bfloat16_as_bitcast(self):
        self.skip_unsupported()

        @cuda.jit
        def roundtrip_kernel(test_val, i2, u2):
            i2[0] = int16_as_bfloat16(bfloat16_as_int16(test_val))
            u2[0] = uint16_as_bfloat16(bfloat16_as_uint16(test_val))

        test_val = np.int16(0x3FC0)  # 1.5 in bfloat16
        i2 = cp.zeros((1,), dtype="int16")
        u2 = cp.zeros((1,), dtype="uint16")
        roundtrip_kernel[1, 1](test_val, i2, u2)

        self.assertEqual(i2[0], test_val)
        self.assertEqual(u2[0], test_val)

    def test_to_integer_conversions(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(test_val, i1, i2, i3, i4, u1, u2, u3, u4):
            a = int16_as_bfloat16(test_val)

            i1[0] = bfloat16_to_int8_rz(a)
            u1[0] = bfloat16_to_uint8_rz(a)
            i2[0] = bfloat16_to_int16_rn(a)
            i2[1] = bfloat16_to_int16_rz(a)
            i2[2] = bfloat16_to_int16_rd(a)
            i2[3] = bfloat16_to_int16_ru(a)
            u2[0] = bfloat16_to_uint16_rn(a)
            u2[1] = bfloat16_to_uint16_rz(a)
            u2[2] = bfloat16_to_uint16_rd(a)
            u2[3] = bfloat16_to_uint16_ru(a)
            i3[0] = bfloat16_to_int32_rn(a)
            i3[1] = bfloat16_to_int32_rz(a)
            i3[2] = bfloat16_to_int32_rd(a)
            i3[3] = bfloat16_to_int32_ru(a)
            u3[0] = bfloat16_to_uint32_rn(a)
            u3[1] = bfloat16_to_uint32_rz(a)
            u3[2] = bfloat16_to_uint32_rd(a)
            u3[3] = bfloat16_to_uint32_ru(a)
            i4[0] = bfloat16_to_int64_rn(a)
            i4[1] = bfloat16_to_int64_rz(a)
            i4[2] = bfloat16_to_int64_rd(a)
            i4[3] = bfloat16_to_int64_ru(a)
            u4[0] = bfloat16_to_uint64_rn(a)
            u4[1] = bfloat16_to_uint64_rz(a)
            u4[2] = bfloat16_to_uint64_rd(a)
            u4[3] = bfloat16_to_uint64_ru(a)

        # rz
        i1 = cp.zeros((1,), dtype="int8")
        # rn, rz, rd, ru
        i2 = cp.zeros((4,), dtype="int16")
        i3 = cp.zeros((4,), dtype="int32")
        i4 = cp.zeros((4,), dtype="int64")
        # rz
        u1 = cp.zeros((1,), dtype="uint8")
        # rn, rz, rd, ru
        u2 = cp.zeros((4,), dtype="uint16")
        u3 = cp.zeros((4,), dtype="uint32")
        u4 = cp.zeros((4,), dtype="uint64")

        test_val = np.int16(0x3FC0)  # 1.5 in bfloat16

        kernel[1, 1](test_val, i1, i2, i3, i4, u1, u2, u3, u4)

        self.assertEqual(i1[0], 1)
        self.assertEqual(u1[0], 1)

        np.testing.assert_equal(i2.get(), np.array([2, 1, 1, 2], "int16"))
        np.testing.assert_equal(i3.get(), np.array([2, 1, 1, 2], "int32"))
        np.testing.assert_equal(i4.get(), np.array([2, 1, 1, 2], "int64"))
        np.testing.assert_equal(u2.get(), np.array([2, 1, 1, 2], "uint16"))
        np.testing.assert_equal(u3.get(), np.array([2, 1, 1, 2], "uint32"))
        np.testing.assert_equal(u4.get(), np.array([2, 1, 1, 2], "uint64"))

    def test_from_integer_conversions(self):
        self.skip_unsupported()

        test_val = 789

        @cuda.jit
        def kernel(out):
            i2 = int16(test_val)
            i3 = int32(test_val)
            i4 = int64(test_val)
            u2 = uint16(test_val)
            u3 = uint32(test_val)
            u4 = uint64(test_val)

            i2rn = int16_to_bfloat16_rn(i2)
            i2rz = int16_to_bfloat16_rz(i2)
            i2rd = int16_to_bfloat16_rd(i2)
            i2ru = int16_to_bfloat16_ru(i2)

            u2rn = uint16_to_bfloat16_rn(u2)
            u2rz = uint16_to_bfloat16_rz(u2)
            u2rd = uint16_to_bfloat16_rd(u2)
            u2ru = uint16_to_bfloat16_ru(u2)

            i3rn = int32_to_bfloat16_rn(i3)
            i3rz = int32_to_bfloat16_rz(i3)
            i3rd = int32_to_bfloat16_rd(i3)
            i3ru = int32_to_bfloat16_ru(i3)

            u3rn = uint32_to_bfloat16_rn(u3)
            u3rz = uint32_to_bfloat16_rz(u3)
            u3rd = uint32_to_bfloat16_rd(u3)
            u3ru = uint32_to_bfloat16_ru(u3)

            i4rn = int64_to_bfloat16_rn(i4)
            i4rz = int64_to_bfloat16_rz(i4)
            i4rd = int64_to_bfloat16_rd(i4)
            i4ru = int64_to_bfloat16_ru(i4)

            u4rn = uint64_to_bfloat16_rn(u4)
            u4rz = uint64_to_bfloat16_rz(u4)
            u4rd = uint64_to_bfloat16_rd(u4)
            u4ru = uint64_to_bfloat16_ru(u4)

            out[0] = bfloat16_as_int16(i2rn)
            out[1] = bfloat16_as_int16(i2rz)
            out[2] = bfloat16_as_int16(i2rd)
            out[3] = bfloat16_as_int16(i2ru)
            out[4] = bfloat16_as_int16(u2rn)
            out[5] = bfloat16_as_int16(u2rz)
            out[6] = bfloat16_as_int16(u2rd)
            out[7] = bfloat16_as_int16(u2ru)
            out[8] = bfloat16_as_int16(i3rn)
            out[9] = bfloat16_as_int16(i3rz)
            out[10] = bfloat16_as_int16(i3rd)
            out[11] = bfloat16_as_int16(i3ru)
            out[12] = bfloat16_as_int16(u3rn)
            out[13] = bfloat16_as_int16(u3rz)
            out[14] = bfloat16_as_int16(u3rd)
            out[15] = bfloat16_as_int16(u3ru)
            out[16] = bfloat16_as_int16(i4rn)
            out[17] = bfloat16_as_int16(i4rz)
            out[18] = bfloat16_as_int16(i4rd)
            out[19] = bfloat16_as_int16(i4ru)
            out[20] = bfloat16_as_int16(u4rn)
            out[21] = bfloat16_as_int16(u4rz)
            out[22] = bfloat16_as_int16(u4rd)
            out[23] = bfloat16_as_int16(u4ru)

        out = cp.zeros((24,), dtype="int16")
        kernel[1, 1](out)
        res = out.get()

        i2 = np.int16(789).astype(mldtypes_bf16).view("int16")
        i3 = np.int32(789).astype(mldtypes_bf16).view("int16")
        i4 = np.int64(789).astype(mldtypes_bf16).view("int16")
        u2 = np.uint16(789).astype(mldtypes_bf16).view("int16")
        u3 = np.uint32(789).astype(mldtypes_bf16).view("int16")
        u4 = np.uint64(789).astype(mldtypes_bf16).view("int16")

        i2arr = np.array([i2] * 4)
        i3arr = np.array([i3] * 4)
        i4arr = np.array([i4] * 4)
        u2arr = np.array([u2] * 4)
        u3arr = np.array([u3] * 4)
        u4arr = np.array([u4] * 4)

        two = np.ones_like(res[0:4]) * 2
        np.testing.assert_array_less(_bf16_ulp_distance(res[0:4], i2arr), two)
        np.testing.assert_array_less(_bf16_ulp_distance(res[4:8], i3arr), two)
        np.testing.assert_array_less(_bf16_ulp_distance(res[8:12], i4arr), two)
        np.testing.assert_array_less(_bf16_ulp_distance(res[12:16], u2arr), two)
        np.testing.assert_array_less(_bf16_ulp_distance(res[16:20], u3arr), two)
        np.testing.assert_array_less(_bf16_ulp_distance(res[20:24], u4arr), two)

    def test_to_float_conversions(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(1.5)
            out[0] = bfloat16_to_float32(a)

        out = cp.zeros((1,), dtype="float32")
        kernel[1, 1](out)

        self.assertAlmostEqual(out[0], 1.5, delta=1e-7)  # conversion is exact

    def test_from_float_conversions(self):
        self.skip_unsupported()

        test_val = 1.5

        @cuda.jit
        def kernel(out):
            f4 = float32(test_val)
            f8 = float64(test_val)

            f4rn = float32_to_bfloat16_rn(f4)
            f4rz = float32_to_bfloat16_rz(f4)
            f4rd = float32_to_bfloat16_rd(f4)
            f4ru = float32_to_bfloat16_ru(f4)

            f4_default = float32_to_bfloat16(f4)
            f8_default = float64_to_bfloat16(f8)

            out[0] = bfloat16_as_int16(f4rn)
            out[1] = bfloat16_as_int16(f4rz)
            out[2] = bfloat16_as_int16(f4rd)
            out[3] = bfloat16_as_int16(f4ru)
            out[4] = bfloat16_as_int16(f4_default)
            out[5] = bfloat16_as_int16(f8_default)

        out = cp.zeros((6,), dtype="int16")
        kernel[1, 1](out)
        raw = out.get()

        f4_expected = (
            np.array([test_val] * 4, "float32")
            .astype(mldtypes_bf16)
            .view("int16")
        )
        f8_expected = (
            np.array([test_val] * 1, "float64")
            .astype(mldtypes_bf16)
            .view("int16")
        )

        np.testing.assert_array_less(
            _bf16_ulp_distance(raw[0:4], f4_expected), 2
        )
        np.testing.assert_array_less(
            _bf16_ulp_distance(raw[4:], f8_expected), 2
        )

    def test_bfloat16_type_import(self):
        self.skip_unsupported()


def _bf16_ulp_rank(bits_int16: np.ndarray) -> np.ndarray:
    """
    Compute the ULP rank of a bfloat16 value. Input is the bits of the bfloat16 value as an int16.
    The ULP rank is the number of ULPs between the value and 0.
    Negative values are performed the inverse of 2's complement before computing the rank.
    """
    u = bits_int16.view(np.uint16)
    sign = u >> 15
    return np.where(sign == 0, u + 0x8000, 0x8000 - u).astype(np.int32)


def _bf16_ulp_distance(
    a_bits_int16: np.ndarray, b_bits_int16: np.ndarray
) -> np.ndarray:
    """
    Compute the difference between two bfloat16 values in ULPs.
    """
    return np.abs(_bf16_ulp_rank(a_bits_int16) - _bf16_ulp_rank(b_bits_int16))
