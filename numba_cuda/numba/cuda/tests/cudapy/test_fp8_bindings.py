# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Comprehensive test suite for CUDA FP8 types and conversion functions.

Tests cover:
- fp8_e5m2, fp8_e4m3, fp8_e8m0 types
- Constructors from various numeric types
- Conversion operators to various types
- Conversion intrinsics
"""

import unittest
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np

from numba.cuda import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)
from numba.cuda.types import float16, bfloat16
from numba.cuda import config
from numba.cuda.api import is_fp8_supported

if not config.ENABLE_CUDASIM:
    from numba.cuda._internal.cuda_fp8 import (
        fp8_e5m2,
        fp8x2_e5m2,
        fp8x4_e5m2,
        fp8_e4m3,
        fp8x2_e4m3,
        fp8x4_e4m3,
        fp8_e8m0,
        fp8x2_e8m0,
        fp8x4_e8m0,
        cvt_float_to_fp8,
        cvt_float2_to_fp8x2,
        cvt_double_to_fp8,
        cvt_double2_to_fp8x2,
        cvt_bfloat16raw_to_fp8,
        cvt_bfloat16raw_to_e8m0,
        cvt_float_to_e8m0,
        cvt_float2_to_e8m0x2,
        cvt_double_to_e8m0,
        cvt_double2_to_e8m0x2,
        cvt_e8m0_to_bf16raw,
        saturation_t,
        fp8_interpretation_t,
    )
    from cuda.bindings.runtime import cudaRoundMode

    FE8_TYPES = [fp8_e5m2, fp8_e4m3, fp8_e8m0]


@unittest.skipUnless(is_fp8_supported(), "FP8 is not supported")
@unittest.skipIf(
    config.ENABLE_CUDASIM, "FP8 is not supported on CUDA simulator"
)
class FP8ConstructorTests(CUDATestCase):
    """Basic constructor for FP8 types."""

    def test_fp8_e5m2_constructors(self):
        """Test fp8_e5m2 construction from all numeric types (floats and ints)."""

        @cuda.jit
        def kernel(result):
            # Test fp8_e5m2 from all float types
            # Use 1.0 which is exactly representable in all FP8 formats
            result[0] = float32(fp8_e5m2(float16(1.0)))
            result[1] = float32(fp8_e5m2(float32(1.0)))
            result[2] = float32(fp8_e5m2(float64(1.0)))
            result[3] = float32(fp8_e5m2(bfloat16(1.0)))

            # Test fp8_e5m2 from all integer types
            # Use 2 which is exactly representable (power of 2)
            result[4] = float32(fp8_e5m2(int8(2)))
            result[5] = float32(fp8_e5m2(int16(2)))
            result[6] = float32(fp8_e5m2(int32(2)))
            result[7] = float32(fp8_e5m2(int64(2)))
            result[8] = float32(fp8_e5m2(uint8(2)))
            result[9] = float32(fp8_e5m2(uint16(2)))
            result[10] = float32(fp8_e5m2(uint32(2)))
            result[11] = float32(fp8_e5m2(uint64(2)))

        result = np.zeros(12, dtype=np.float32)
        kernel[1, 1](result)

        # Check float conversions (indices 0-3) - should be exactly 1.0
        for i in range(4):
            self.assertEqual(result[i], 1.0)

        # Check integer conversions (indices 4-11) - should be exactly 2.0
        for i in range(4, 12):
            self.assertEqual(result[i], 2.0)

    def test_fp8_e4m3_constructors(self):
        """Test fp8_e4m3 construction from all numeric types (floats and ints)."""

        @cuda.jit
        def kernel(result):
            # Test fp8_e4m3 from all float types
            # Use 1.0 which is exactly representable in all FP8 formats
            result[0] = float32(fp8_e4m3(float16(1.0)))
            result[1] = float32(fp8_e4m3(float32(1.0)))
            result[2] = float32(fp8_e4m3(float64(1.0)))
            result[3] = float32(fp8_e4m3(bfloat16(1.0)))

            # Test fp8_e4m3 from all integer types
            # Use 2 which is exactly representable (power of 2)
            result[4] = float32(fp8_e4m3(int8(2)))
            result[5] = float32(fp8_e4m3(int16(2)))
            result[6] = float32(fp8_e4m3(int32(2)))
            result[7] = float32(fp8_e4m3(int64(2)))
            result[8] = float32(fp8_e4m3(uint8(2)))
            result[9] = float32(fp8_e4m3(uint16(2)))
            result[10] = float32(fp8_e4m3(uint32(2)))
            result[11] = float32(fp8_e4m3(uint64(2)))

        result = np.zeros(12, dtype=np.float32)
        kernel[1, 1](result)

        # Check float conversions (indices 0-3) - should be exactly 1.0
        for i in range(4):
            self.assertEqual(result[i], 1.0)

        # Check integer conversions (indices 4-11) - should be exactly 2.0
        for i in range(4, 12):
            self.assertEqual(result[i], 2.0)

    def test_fp8_e8m0_constructors(self):
        """Test fp8_e8m0 construction from all numeric types (floats and ints)."""

        @cuda.jit
        def kernel(result):
            # Test fp8_e8m0 from all float types (use power of 2 for e8m0)
            result[0] = float32(fp8_e8m0(float16(2.0)))
            result[1] = float32(fp8_e8m0(float32(2.0)))
            result[2] = float32(fp8_e8m0(float64(2.0)))
            result[3] = float32(fp8_e8m0(bfloat16(2.0)))

            # Test fp8_e8m0 from all integer types (use power of 2)
            result[4] = float32(fp8_e8m0(int8(4)))
            result[5] = float32(fp8_e8m0(int16(4)))
            result[6] = float32(fp8_e8m0(int32(4)))
            result[7] = float32(fp8_e8m0(int64(4)))
            result[8] = float32(fp8_e8m0(uint8(4)))
            result[9] = float32(fp8_e8m0(uint16(4)))
            result[10] = float32(fp8_e8m0(uint32(4)))
            result[11] = float32(fp8_e8m0(uint64(4)))

        result = np.zeros(12, dtype=np.float32)
        kernel[1, 1](result)

        # Check float conversions - should be exactly 2.0 (e8m0 stores powers of 2)
        for i in range(4):
            self.assertEqual(result[i], 2.0)

        # Check integer conversions - should be exactly 4.0
        for i in range(4, 12):
            self.assertEqual(result[i], 4.0)

    def test_fp8x2_constructors_match_conversion_intrinsics(self):
        """Test fp8x2 packed constructors from float2/double2 inputs."""

        @cuda.jit
        def kernel(result):
            f2 = cuda.float32x2(float32(1.0), float32(2.0))
            d2 = cuda.float64x2(float64(4.0), float64(8.0))

            result[0] = fp8x2_e5m2(f2).__x
            result[1] = cvt_float2_to_fp8x2(
                f2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            result[2] = fp8x2_e4m3(f2).__x
            result[3] = cvt_float2_to_fp8x2(
                f2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            result[4] = fp8x2_e8m0(f2).__x
            result[5] = cvt_float2_to_e8m0x2(
                f2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            result[6] = fp8x2_e5m2(d2).__x
            result[7] = cvt_double2_to_fp8x2(
                d2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            result[8] = fp8x2_e4m3(d2).__x
            result[9] = cvt_double2_to_fp8x2(
                d2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            result[10] = fp8x2_e8m0(d2).__x
            result[11] = cvt_double2_to_e8m0x2(
                d2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

        result = np.zeros(12, dtype=np.uint16)
        kernel[1, 1](result)

        for i in range(0, 12, 2):
            self.assertEqual(result[i], result[i + 1])

    def test_fp8x4_constructors_match_paired_x2_intrinsics(self):
        """Test fp8x4 packed constructors from float4/double4 inputs."""

        @cuda.jit
        def kernel(result):
            f4 = cuda.float32x4(
                float32(1.0), float32(2.0), float32(4.0), float32(8.0)
            )
            d4 = cuda.float64x4(
                float64(16.0), float64(32.0), float64(64.0), float64(128.0)
            )

            f_lo2 = cuda.float32x2(f4.x, f4.y)
            f_hi2 = cuda.float32x2(f4.z, f4.w)
            d_lo2 = cuda.float64x2(d4.x, d4.y)
            d_hi2 = cuda.float64x2(d4.z, d4.w)

            e5m2_lo_f = cvt_float2_to_fp8x2(
                f_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            e5m2_hi_f = cvt_float2_to_fp8x2(
                f_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            e4m3_lo_f = cvt_float2_to_fp8x2(
                f_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            e4m3_hi_f = cvt_float2_to_fp8x2(
                f_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            e8m0_lo_f = cvt_float2_to_e8m0x2(
                f_lo2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )
            e8m0_hi_f = cvt_float2_to_e8m0x2(
                f_hi2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            e5m2_lo_d = cvt_double2_to_fp8x2(
                d_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            e5m2_hi_d = cvt_double2_to_fp8x2(
                d_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            e4m3_lo_d = cvt_double2_to_fp8x2(
                d_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            e4m3_hi_d = cvt_double2_to_fp8x2(
                d_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            e8m0_lo_d = cvt_double2_to_e8m0x2(
                d_lo2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )
            e8m0_hi_d = cvt_double2_to_e8m0x2(
                d_hi2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            result[0] = fp8x4_e5m2(f4).__x
            result[1] = uint32(e5m2_lo_f) | (uint32(e5m2_hi_f) << 16)
            result[2] = fp8x4_e4m3(f4).__x
            result[3] = uint32(e4m3_lo_f) | (uint32(e4m3_hi_f) << 16)
            result[4] = fp8x4_e8m0(f4).__x
            result[5] = uint32(e8m0_lo_f) | (uint32(e8m0_hi_f) << 16)

            result[6] = fp8x4_e5m2(d4).__x
            result[7] = uint32(e5m2_lo_d) | (uint32(e5m2_hi_d) << 16)
            result[8] = fp8x4_e4m3(d4).__x
            result[9] = uint32(e4m3_lo_d) | (uint32(e4m3_hi_d) << 16)
            result[10] = fp8x4_e8m0(d4).__x
            result[11] = uint32(e8m0_lo_d) | (uint32(e8m0_hi_d) << 16)

        result = np.zeros(12, dtype=np.uint32)
        kernel[1, 1](result)

        for i in range(0, 12, 2):
            self.assertEqual(result[i], result[i + 1])

    def test_fp8_nan_constructors(self):
        """Test fp8 construction from NaN."""

        @cuda.jit
        def kernel(result):
            nan = float32(float("nan"))
            result[0] = float32(fp8_e5m2(nan))
            result[1] = float32(fp8_e4m3(nan))
            result[2] = float32(fp8_e8m0(nan))

        result = np.zeros(3, dtype=np.float32)
        kernel[1, 1](result)

        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))


@unittest.skipUnless(is_fp8_supported(), "FP8 is not supported")
@unittest.skipIf(
    config.ENABLE_CUDASIM, "FP8 is not supported on CUDA simulator"
)
class FP8ConversionTests(CUDATestCase):
    """Test FP8 conversion operators to various types."""

    def test_fp8_to_float_types(self):
        """Test FP8 conversion to float types (__half, float, double, bfloat16)."""

        # Test data: (fp8_type, test_value, expected, tolerance_places)
        test_cases = [
            (fp8_e5m2, 1.5, 1.5, 2),
            (fp8_e4m3, 2.5, 2.5, 2),
            (fp8_e8m0, 8.0, 8.0, None),  # Power of 2, exact
        ]

        for fp8_type, test_val, expected, places in test_cases:
            with self.subTest(fp8_type=fp8_type.__name__, value=test_val):

                @cuda.jit
                def kernel(result):
                    fp8_val = fp8_type(float32(test_val))

                    result[0] = float32(float16(fp8_val))  # to __half
                    result[1] = float32(fp8_val)  # to float
                    result[2] = float32(float64(fp8_val))  # to double
                    result[3] = float32(bfloat16(fp8_val))  # to bfloat16

                result = np.zeros(4, dtype=np.float32)
                kernel[1, 1](result)

                if places is None:
                    self.assertTrue(np.all(result == expected))
                else:
                    self.assertTrue(
                        np.allclose(
                            result, expected, rtol=0, atol=10 ** (-places)
                        )
                    )

    def test_fp8_to_unsigned_integers(self):
        """Test FP8 conversion to unsigned integer types."""

        # Test data: (fp8_type, test_value, expected)
        test_cases = [
            (fp8_e5m2, 10.0, 10),
            (fp8_e4m3, 12.0, 12),
            (fp8_e8m0, 16.0, 16),
        ]

        for fp8_type, test_val, expected in test_cases:
            with self.subTest(fp8_type=fp8_type.__name__, value=test_val):

                @cuda.jit
                def kernel(result):
                    fp8_val = fp8_type(float32(test_val))

                    result[0] = uint64(uint8(fp8_val))
                    result[1] = uint64(uint16(fp8_val))
                    result[2] = uint64(uint32(fp8_val))
                    result[3] = uint64(fp8_val)

                result = np.zeros(4, dtype=np.uint64)
                kernel[1, 1](result)

                self.assertTrue(np.all(result == expected))

    def test_fp8_to_signed_integers(self):
        """Test FP8 conversion to signed integer types."""

        # Test data: (fp8_type, positive_value, negative_value)
        test_cases = [
            (fp8_e5m2, 16.0, -8.0),
            (fp8_e4m3, 20.0, -6.0),
            (
                fp8_e8m0,
                32.0,
                16.0,
            ),  # Because fp8_e8m0 is an exponent only type, it does not represent negative values.
        ]

        for fp8_type, pos_val, neg_val in test_cases:
            with self.subTest(fp8_type=fp8_type.__name__):

                @cuda.jit
                def kernel(result):
                    fp8_pos = fp8_type(float32(pos_val))
                    fp8_neg = fp8_type(float32(neg_val))

                    result[0] = int64(int8(fp8_pos))
                    result[1] = int64(int16(fp8_pos))
                    result[2] = int64(int32(fp8_pos))
                    result[3] = int64(fp8_pos)
                    result[4] = int64(int8(fp8_neg))
                    result[5] = int64(int16(fp8_neg))
                    result[6] = int64(int32(fp8_neg))
                    result[7] = int64(fp8_neg)

                result = np.zeros(8, dtype=np.int64)
                kernel[1, 1](result)

                # Check positive conversions
                np.testing.assert_array_equal(
                    result[:4], np.array([int(pos_val)] * 4)
                )

                # Check negative conversions
                np.testing.assert_array_equal(
                    result[4:], np.array([int(neg_val)] * 4)
                )

    def test_fp8_conversion_edge_cases_zero(self):
        """Test conversion of zero values for all FP8 types."""

        for fp8_type in FE8_TYPES:
            with self.subTest(fp8_type=fp8_type.__name__):

                @cuda.jit
                def kernel(result):
                    zero = fp8_type(float32(0.0))
                    result[0] = float32(zero)
                    result[1] = int64(zero)

                result = np.zeros(2, dtype=np.int64)
                kernel[1, 1](result)

                np.testing.assert_array_equal(result, np.array([0, 0]))

    def test_fp8_conversion_negative_values(self):
        """Test conversion of negative values for all FP8 types.
        Because fp8_e8m0 is an exponent only type, it does not represent negative values.
        """

        # Test data: (fp8_type, test_value)
        test_cases = [(fp8_e5m2, -3.0), (fp8_e4m3, -4.0)]

        for fp8_type, neg_val in test_cases:
            with self.subTest(fp8_type=fp8_type.__name__, value=neg_val):

                @cuda.jit
                def kernel(result_float, result_int):
                    fp8_neg = fp8_type(float32(neg_val))

                    result_float[0] = float32(fp8_neg)
                    result_int[0] = int64(int32(fp8_neg))

                result_float = np.zeros(1, dtype=np.float32)
                result_int = np.zeros(1, dtype=np.int64)
                kernel[1, 1](result_float, result_int)

                self.assertAlmostEqual(result_float[0], neg_val, places=1)
                self.assertAlmostEqual(result_int[0], int(neg_val), delta=1)

    def test_fp8_conversion_roundtrip(self):
        """Test roundtrip conversions: fp8 -> float -> fp8 -> float."""

        for fp8_type in FE8_TYPES:
            with self.subTest(fp8_type=fp8_type.__name__):

                @cuda.jit
                def kernel(result):
                    float_val = float32(4.0)
                    fp8_val2 = fp8_type(float_val)
                    result[0] = float32(fp8_val2)

                result = np.zeros(1, dtype=np.float32)
                kernel[1, 1](result)

                np.testing.assert_array_equal(result, np.array([4.0]))

    def test_fp8_nan_conversions(self):
        """Test FP8 NaN conversion to other types."""

        @cuda.jit
        def kernel(result):
            nan = float32(float("nan"))
            # Create FP8 NaNs
            nan_e5m2 = fp8_e5m2(nan)
            nan_e4m3 = fp8_e4m3(nan)
            nan_e8m0 = fp8_e8m0(nan)

            # Convert back to float types
            result[0] = float32(float16(nan_e5m2))
            result[1] = float32(nan_e5m2)
            result[2] = float32(float64(nan_e5m2))

            result[3] = float32(float16(nan_e4m3))
            result[4] = float32(nan_e4m3)
            result[5] = float32(float64(nan_e4m3))

            result[6] = float32(float16(nan_e8m0))
            result[7] = float32(nan_e8m0)
            result[8] = float32(float64(nan_e8m0))

        result = np.zeros(9, dtype=np.float32)
        kernel[1, 1](result)

        np.testing.assert_array_equal(result, np.array([float("nan")] * 9))


@unittest.skipUnless(is_fp8_supported(), "FP8 is not supported")
@unittest.skipIf(
    config.ENABLE_CUDASIM, "FP8 is not supported on CUDA simulator"
)
class FP8Storage_CVT_Intrinsics_Tests(CUDATestCase):
    """Test raw conversion intrinsics operating on storage types."""

    def test_cvt_float_to_fp8(self):
        @cuda.jit
        def kernel(result, x):
            # Use an out-of-range value so NOSAT and SATFINITE differ.
            # For overflow:
            # - E5M2: NOSAT -> Inf (0x7C), SATFINITE -> MAXNORM (0x7B)
            # - E4M3: NOSAT -> NaN (0x7F), SATFINITE -> MAXNORM (0x7E)
            result[0] = cvt_float_to_fp8(
                x[0], saturation_t.NOSAT, fp8_interpretation_t.E5M2
            )
            result[1] = cvt_float_to_fp8(
                x[0], saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            result[2] = cvt_float_to_fp8(
                x[0], saturation_t.NOSAT, fp8_interpretation_t.E4M3
            )
            result[3] = cvt_float_to_fp8(
                x[0], saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )

        result = np.zeros(4, dtype=np.uint8)
        x = np.array([1e20], dtype=np.float32)
        kernel[1, 1](result, x)

        self.assertEqual(result[0], 0x7C)  # E5M2 overflow -> Inf (NOSAT)
        self.assertEqual(
            result[1], 0x7B
        )  # E5M2 overflow -> MAXNORM (SATFINITE)
        self.assertEqual(result[2], 0x7F)  # E4M3 overflow -> NaN (NOSAT)
        self.assertEqual(
            result[3], 0x7E
        )  # E4M3 overflow -> MAXNORM (SATFINITE)

    def test_cvt_double_to_fp8(self):
        @cuda.jit
        def kernel(result, x):
            result[0] = cvt_double_to_fp8(
                x[0], saturation_t.NOSAT, fp8_interpretation_t.E5M2
            )
            result[1] = cvt_double_to_fp8(
                x[0], saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            result[2] = cvt_double_to_fp8(
                x[0], saturation_t.NOSAT, fp8_interpretation_t.E4M3
            )
            result[3] = cvt_double_to_fp8(
                x[0], saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )

        result = np.zeros(4, dtype=np.uint8)
        x = np.array([1e300], dtype=np.float64)
        kernel[1, 1](result, x)
        self.assertEqual(result[0], 0x7C)  # E5M2 overflow -> Inf (NOSAT)
        self.assertEqual(
            result[1], 0x7B
        )  # E5M2 overflow -> MAXNORM (SATFINITE)
        self.assertEqual(result[2], 0x7F)  # E4M3 overflow -> NaN (NOSAT)
        self.assertEqual(
            result[3], 0x7E
        )  # E4M3 overflow -> MAXNORM (SATFINITE)

    def test_cvt_e8m0_to_bf16raw(self):
        @cuda.jit
        def kernel(result, x):
            raw = cvt_e8m0_to_bf16raw(x[0])
            result[0] = raw.x

        result = np.zeros(1, dtype=np.uint16)
        # 1.0 in E8M0 is 127 (bias 127)
        x = np.array([127], dtype=np.uint8)
        kernel[1, 1](result, x)

        # 1.0 in BF16 is 0x3F80
        self.assertEqual(result[0], 0x3F80)

    def test_cvt_bfloat16raw_roundtrip(self):
        @cuda.jit
        def kernel(result, x):
            # x is uint8 (e8m0)
            # Convert e8m0 to bfloat16_raw
            raw = cvt_e8m0_to_bf16raw(x[0])

            # Convert bf16_raw to fp8 using both NOSAT and SATFINITE.
            result[0] = cvt_bfloat16raw_to_fp8(
                raw, saturation_t.NOSAT, fp8_interpretation_t.E5M2
            )
            result[1] = cvt_bfloat16raw_to_fp8(
                raw, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            result[2] = cvt_bfloat16raw_to_fp8(
                raw, saturation_t.NOSAT, fp8_interpretation_t.E4M3
            )
            result[3] = cvt_bfloat16raw_to_fp8(
                raw, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )

        result = np.zeros(4, dtype=np.uint8)
        # 2^127 in E8M0 (very out-of-range for fp8)
        x = np.array([254], dtype=np.uint8)
        kernel[1, 1](result, x)

        self.assertEqual(result[0], 0x7C)  # E5M2 overflow -> Inf (NOSAT)
        self.assertEqual(
            result[1], 0x7B
        )  # E5M2 overflow -> MAXNORM (SATFINITE)
        self.assertEqual(result[2], 0x7F)  # E4M3 overflow -> NaN (NOSAT)
        self.assertEqual(
            result[3], 0x7E
        )  # E4M3 overflow -> MAXNORM (SATFINITE)

    def test_cvt_float_to_e8m0(self):
        @cuda.jit
        def kernel(result, x):
            # Use a value slightly larger than 2^127 and round up so the
            # rounded scale would overflow. Then SATFINITE clips to 2^127 while
            # NOSAT produces NaN (0xFF).
            result[0] = cvt_float_to_e8m0(
                x[0], saturation_t.NOSAT, cudaRoundMode.cudaRoundPosInf
            )
            result[1] = cvt_float_to_e8m0(
                x[0], saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

        x_over = np.nextafter(np.float32(2.0**127), np.float32(np.inf))
        x = np.array([x_over], dtype=np.float32)
        result = np.zeros(2, dtype=np.uint8)
        kernel[1, 1](result, x)

        self.assertEqual(result[0], 0xFF)  # NOSAT overflow -> NaN
        self.assertEqual(
            result[1], 0xFE
        )  # SATFINITE overflow -> max finite (2^127)

    def test_cvt_double_to_e8m0(self):
        @cuda.jit
        def kernel(result, x):
            result[0] = cvt_double_to_e8m0(
                x[0], saturation_t.NOSAT, cudaRoundMode.cudaRoundPosInf
            )
            result[1] = cvt_double_to_e8m0(
                x[0], saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

        x_over = np.nextafter(np.float64(2.0**127), np.float64(np.inf))
        x = np.array([x_over], dtype=np.float64)
        result = np.zeros(2, dtype=np.uint8)
        kernel[1, 1](result, x)

        self.assertEqual(result[0], 0xFF)  # NOSAT overflow -> NaN
        self.assertEqual(
            result[1], 0xFE
        )  # SATFINITE overflow -> max finite (2^127)

    def test_cvt_bfloat16raw_to_e8m0(self):
        @cuda.jit
        def kernel(result, exponent):
            # NOTE: We currently don't have a good way to construct/modify a
            # bfloat16_raw value directly in kernels, so we skip SATFINITE vs
            # NOSAT behavior testing for bf16raw here.
            raw = cvt_e8m0_to_bf16raw(exponent[0])
            result[0] = cvt_bfloat16raw_to_e8m0(
                raw, saturation_t.NOSAT, cudaRoundMode.cudaRoundZero
            )
            result[1] = cvt_bfloat16raw_to_e8m0(
                raw, saturation_t.NOSAT, cudaRoundMode.cudaRoundPosInf
            )

        exponent = np.array([126], dtype=np.uint8)  # 0.5
        result = np.zeros(2, dtype=np.uint8)
        kernel[1, 1](result, exponent)

        self.assertEqual(result[0], 126)
        self.assertEqual(result[1], 126)


if __name__ == "__main__":
    unittest.main()
