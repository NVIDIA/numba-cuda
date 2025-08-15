import unittest
from importlib.util import find_spec

from numba import cuda, float32
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
    bfloat16_to_float32,
    float32_to_bfloat16,
    float64_to_bfloat16,
    float32_to_bfloat16_rn,
    float32_to_bfloat16_rz,
    float32_to_bfloat16_rd,
    float32_to_bfloat16_ru,
    int32_to_bfloat16_rn,
    int32_to_bfloat16_rz,
    int32_to_bfloat16_rd,
    int32_to_bfloat16_ru,
    bfloat16_to_int32_rn,
    bfloat16_to_int32_rz,
    bfloat16_to_int32_rd,
    bfloat16_to_int32_ru,
    bfloat16_to_int16_rn,
    int16_to_bfloat16_rn,
    bfloat16_to_uint16_rn,
    uint16_to_bfloat16_rn,
    bfloat16_to_uint32_rn,
    uint32_to_bfloat16_rn,
    bfloat16_to_int64_rn,
    int64_to_bfloat16_rn,
    bfloat16_to_uint64_rn,
    uint64_to_bfloat16_rn,
    bfloat16_as_short,
    bfloat16_as_ushort,
    short_as_bfloat16,
    ushort_as_bfloat16,
    bfloat16_to_int8_rz,
    bfloat16_to_uint8_rz,
)
from numba.cuda.testing import CUDATestCase

import math


class TestBfloat16HighLevelBindings(CUDATestCase):
    def skip_unsupported(self):
        if not cuda.is_bfloat16_supported():
            self.skipTest(
                "bfloat16 requires compute capability 8.0+ and CUDA version>= 12.0"
            )

    def test_use_type_in_kernel(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel():
            bfloat16(3.14)

        kernel[1, 1]()

    def test_math_bindings(self):
        self.skip_unsupported()

        exp_functions = [math.exp]
        try:
            from math import exp2

            exp_functions += [exp2]
        except ImportError:
            pass

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

                arr = cuda.device_array((1,), dtype="float32")
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

        out = cuda.device_array((10,), dtype="float32")
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

        out = cuda.device_array((4,), dtype="float32")
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

        out = cuda.device_array((1,), dtype="float32")
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
                out = cuda.device_array((1,), dtype="bool")

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

        out = cuda.device_array((2,), dtype="float32")
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

        out_bool = cuda.device_array((1,), dtype="bool")
        out_int = cuda.device_array((1,), dtype="int32")
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

        out = cuda.device_array((4,), dtype="float32")
        kernel[1, 1](out)
        # NaN-propagating variants should produce NaN
        self.assertTrue(math.isnan(out[0]))
        self.assertTrue(math.isnan(out[1]))
        # Non-NaN variants should return the non-NaN operand
        self.assertAlmostEqual(out[2], 2.0, delta=1e-3)
        self.assertAlmostEqual(out[3], 2.0, delta=1e-3)

    def test_int32_float32_precision_conversion_intrinsics(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel_float_to_bf16(out):
            f = float32(3.14)
            out[0] = float32(float32_to_bfloat16_rn(f))
            out[1] = float32(float32_to_bfloat16_rz(f))
            out[2] = float32(float32_to_bfloat16_rd(f))
            out[3] = float32(float32_to_bfloat16_ru(f))

        @cuda.jit
        def kernel_bf16_to_float(out):
            a = bfloat16(3.14)
            out[0] = bfloat16_to_float32(a)

        @cuda.jit
        def kernel_int_to_bf16(out):
            i = 3
            out[0] = float32(int32_to_bfloat16_rn(i))
            out[1] = float32(int32_to_bfloat16_rz(i))
            out[2] = float32(int32_to_bfloat16_rd(i))
            out[3] = float32(int32_to_bfloat16_ru(i))

        @cuda.jit
        def kernel_bf16_to_int(out):
            a = bfloat16(3.14)
            out[0] = bfloat16_to_int32_rn(a)
            out[1] = bfloat16_to_int32_rz(a)
            out[2] = bfloat16_to_int32_rd(a)
            out[3] = bfloat16_to_int32_ru(a)

        out = cuda.device_array((4,), dtype="float32")
        kernel_float_to_bf16[1, 1](out)
        # Check they are near the original value in float32 after round-trip
        # Note: Different rounding modes produce slightly different values
        self.assertAlmostEqual(out[0], 3.140625, delta=1e-3)  # rn
        self.assertTrue(abs(out[1] - 3.140625) < 2e-2, out[1] - 3.140625)  # rz
        self.assertTrue(abs(out[2] - 3.140625) < 2e-2, out[2] - 3.140625)  # rd
        self.assertTrue(abs(out[3] - 3.140625) < 2e-2, out[3] - 3.140625)  # ru

        out = cuda.device_array((1,), dtype="float32")
        kernel_bf16_to_float[1, 1](out)
        self.assertAlmostEqual(out[0], 3.140625, delta=1e-3)

        outi = cuda.device_array((4,), dtype="int32")
        kernel_int_to_bf16[1, 1](outi)
        # int to bf16 should be exactly representable for small integers
        self.assertEqual(int(outi[0]), 3)
        self.assertEqual(int(outi[1]), 3)
        self.assertEqual(int(outi[2]), 3)
        self.assertEqual(int(outi[3]), 3)

        outi = cuda.device_array((4,), dtype="int32")
        kernel_bf16_to_int[1, 1](outi)
        # 3.14 -> 3 for rz/rd, 3 or 4 for rn/ru depending on rounding
        self.assertIn(int(outi[0]), (3, 4))
        self.assertEqual(int(outi[1]), 3)
        self.assertEqual(int(outi[2]), 3)
        self.assertIn(int(outi[3]), (3, 4))

    def test_floatroundtrip_integer_conversion_intrinsics(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel_scalar_roundtrip(out):
            f = 3.14
            bf = float32_to_bfloat16(f)
            out[0] = bfloat16_to_float32(bf)
            d = 3.14
            bf2 = float64_to_bfloat16(d)
            out[1] = bfloat16_to_float32(bf2)

        out = cuda.device_array((2,), dtype="float32")
        kernel_scalar_roundtrip[1, 1](out)
        self.assertAlmostEqual(out[0], 3.140625, delta=1e-3)
        self.assertAlmostEqual(out[1], 3.140625, delta=1e-3)

        @cuda.jit
        def kernel_int_family(outf):
            outf[0] = float32(int16_to_bfloat16_rn(123))
            outf[1] = float32(uint16_to_bfloat16_rn(456))
            outf[2] = float32(uint32_to_bfloat16_rn(789))
            outf[3] = float32(int64_to_bfloat16_rn(1011))
            outf[4] = float32(uint64_to_bfloat16_rn(1213))

        outf = cuda.device_array((5,), dtype="float32")
        kernel_int_family[1, 1](outf)
        vals = [123, 456, 789, 1011, 1213]
        for i, v in enumerate(vals):
            got = int(outf[i])
            # `step` estimates ULP near the integer `v`.
            # Bfloat16 has 7 bits of precision, spacing between representable values are 2**(e-7).
            # We use the exponent of the value `v` to raise the minSpacing, the result is a reasonable
            # esitmate the local ULP.
            step = (
                0 if v == 0 else 2 ** (int(math.floor(math.log2(abs(v)))) - 7)
            )
            # `allowed` is the maximum error in ULP, with a minimum of 1
            # In general, half ULP is the typical rounding error bound.
            allowed = max(1, int(step // 2))
            self.assertLessEqual(abs(got - v), allowed)

        @cuda.jit
        def kernel_from_bf16_to_ints(outi):
            a = bfloat16(5.75)
            outi[0] = bfloat16_to_int16_rn(a)
            outi[1] = bfloat16_to_uint16_rn(a)
            outi[2] = bfloat16_to_uint32_rn(a)
            outi[3] = bfloat16_to_int64_rn(a)
            outi[4] = bfloat16_to_uint64_rn(a)

        outi = cuda.device_array((5,), dtype="int64")
        kernel_from_bf16_to_ints[1, 1](outi)
        self.assertEqual(int(outi[0]), 6)
        self.assertEqual(int(outi[1]), 6)
        self.assertEqual(int(outi[2]), 6)
        self.assertEqual(int(outi[3]), 6)
        self.assertEqual(int(outi[4]), 6)

        @cuda.jit
        def kernel_bit_reinterpret(out_short, out_ushort):
            s = 12345
            bf = short_as_bfloat16(s)
            out_short[0] = bfloat16_as_short(bf)
            us = 54321
            bf2 = ushort_as_bfloat16(us)
            out_ushort[0] = bfloat16_as_ushort(bf2)

        out_short = cuda.device_array((1,), dtype="int32")
        out_ushort = cuda.device_array((1,), dtype="uint32")
        kernel_bit_reinterpret[1, 1](out_short, out_ushort)
        self.assertEqual(int(out_short[0]), 12345)
        self.assertEqual(int(out_ushort[0]), 54321)

        @cuda.jit
        def kernel_char(out_c, out_uc):
            a = bfloat16(3.9)
            out_c[0] = bfloat16_to_int8_rz(a)
            out_uc[0] = bfloat16_to_uint8_rz(a)

        out_c = cuda.device_array((1,), dtype="int8")
        out_uc = cuda.device_array((1,), dtype="uint8")
        kernel_char[1, 1](out_c, out_uc)
        self.assertEqual(int(out_c[0]), 3)
        self.assertEqual(int(out_uc[0]), 3)

    @unittest.skipIf(
        find_spec("ml_dtypes") is None,
        "ml_dtypes is required to use bfloat16 on host",
    )
    def test_use_bfloat16_on_host(self):
        x = bfloat16(3.0)
        self.assertEqual(x, 3.0)
