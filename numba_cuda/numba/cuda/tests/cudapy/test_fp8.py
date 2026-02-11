# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import config, float32, float64, uint8, uint32
from numba.cuda.api import is_fp8_supported
from numba.cuda.testing import CUDATestCase, unittest

if not config.ENABLE_CUDASIM:
    from cuda.bindings.runtime import cudaRoundMode

    from numba.cuda.fp8 import (
        bfloat16_raw_to_e8m0,
        bfloat16_raw_to_fp8,
        cvt_double2_to_e8m0x2,
        cvt_double2_to_fp8x2,
        cvt_double_to_fp8,
        cvt_float2_to_e8m0x2,
        cvt_float2_to_fp8x2,
        cvt_float_to_fp8,
        e8m0_to_bfloat16_raw,
        float32_to_e8m0,
        float32_to_fp8,
        float32x2_to_e8m0x2,
        float32x2_to_fp8x2,
        float64_to_e8m0,
        float64_to_fp8,
        float64x2_to_e8m0x2,
        float64x2_to_fp8x2,
        fp8_e4m3,
        fp8_e5m2,
        fp8_e8m0,
        fp8x2_e4m3,
        fp8x2_e5m2,
        fp8x2_e8m0,
        fp8x4_e4m3,
        fp8x4_e5m2,
        fp8x4_e8m0,
        fp8_interpretation_t,
        saturation_t,
    )


@unittest.skipUnless(is_fp8_supported(), "FP8 is not supported")
@unittest.skipIf(
    config.ENABLE_CUDASIM, "FP8 is not supported on CUDA simulator"
)
class TestFP8HighLevelBindings(CUDATestCase):
    def test_public_aliases_map_to_intrinsics(self):
        self.assertIs(float32_to_fp8, cvt_float_to_fp8)
        self.assertIs(float64_to_fp8, cvt_double_to_fp8)
        self.assertIs(float32x2_to_fp8x2, cvt_float2_to_fp8x2)
        self.assertIs(float64x2_to_fp8x2, cvt_double2_to_fp8x2)
        self.assertIs(float32x2_to_e8m0x2, cvt_float2_to_e8m0x2)
        self.assertIs(float64x2_to_e8m0x2, cvt_double2_to_e8m0x2)

    def test_scalar_fp8_types(self):
        @cuda.jit
        def kernel(out):
            out[0] = float32(fp8_e5m2(float32(1.0)))
            out[1] = float32(fp8_e4m3(float32(1.0)))
            out[2] = float32(fp8_e8m0(float32(2.0)))

        out = np.zeros(3, dtype=np.float32)
        kernel[1, 1](out)

        np.testing.assert_allclose(out, [1.0, 1.0, 2.0], atol=1e-3)

    def test_packed_fp8_types(self):
        @cuda.jit
        def kernel(out_u16, out_u32):
            f2 = cuda.float32x2(float32(1.0), float32(2.0))
            d2 = cuda.float64x2(float64(4.0), float64(8.0))

            out_u16[0] = fp8x2_e5m2(f2).__x
            out_u16[1] = float32x2_to_fp8x2(
                f2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            out_u16[2] = fp8x2_e4m3(f2).__x
            out_u16[3] = float32x2_to_fp8x2(
                f2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            out_u16[4] = fp8x2_e8m0(f2).__x
            out_u16[5] = float32x2_to_e8m0x2(
                f2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            out_u16[6] = fp8x2_e5m2(d2).__x
            out_u16[7] = float64x2_to_fp8x2(
                d2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            out_u16[8] = fp8x2_e4m3(d2).__x
            out_u16[9] = float64x2_to_fp8x2(
                d2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            out_u16[10] = fp8x2_e8m0(d2).__x
            out_u16[11] = float64x2_to_e8m0x2(
                d2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

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

            f_e5m2_lo = float32x2_to_fp8x2(
                f_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            f_e5m2_hi = float32x2_to_fp8x2(
                f_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            f_e4m3_lo = float32x2_to_fp8x2(
                f_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            f_e4m3_hi = float32x2_to_fp8x2(
                f_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            f_e8m0_lo = float32x2_to_e8m0x2(
                f_lo2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )
            f_e8m0_hi = float32x2_to_e8m0x2(
                f_hi2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            d_e5m2_lo = float64x2_to_fp8x2(
                d_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            d_e5m2_hi = float64x2_to_fp8x2(
                d_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            d_e4m3_lo = float64x2_to_fp8x2(
                d_lo2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            d_e4m3_hi = float64x2_to_fp8x2(
                d_hi2, saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )
            d_e8m0_lo = float64x2_to_e8m0x2(
                d_lo2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )
            d_e8m0_hi = float64x2_to_e8m0x2(
                d_hi2, saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            out_u32[0] = fp8x4_e5m2(f4).__x
            out_u32[1] = uint32(f_e5m2_lo) | (uint32(f_e5m2_hi) << 16)
            out_u32[2] = fp8x4_e4m3(f4).__x
            out_u32[3] = uint32(f_e4m3_lo) | (uint32(f_e4m3_hi) << 16)
            out_u32[4] = fp8x4_e8m0(f4).__x
            out_u32[5] = uint32(f_e8m0_lo) | (uint32(f_e8m0_hi) << 16)

            out_u32[6] = fp8x4_e5m2(d4).__x
            out_u32[7] = uint32(d_e5m2_lo) | (uint32(d_e5m2_hi) << 16)
            out_u32[8] = fp8x4_e4m3(d4).__x
            out_u32[9] = uint32(d_e4m3_lo) | (uint32(d_e4m3_hi) << 16)
            out_u32[10] = fp8x4_e8m0(d4).__x
            out_u32[11] = uint32(d_e8m0_lo) | (uint32(d_e8m0_hi) << 16)

        out_u16 = np.zeros(12, dtype=np.uint16)
        out_u32 = np.zeros(12, dtype=np.uint32)
        kernel[1, 1](out_u16, out_u32)

        for i in range(0, 12, 2):
            self.assertEqual(out_u16[i], out_u16[i + 1])
            self.assertEqual(out_u32[i], out_u32[i + 1])

    def test_float_to_fp8_aliases(self):
        @cuda.jit
        def kernel(out_u8, x32, x64):
            out_u8[0] = float32_to_fp8(
                x32[0], saturation_t.NOSAT, fp8_interpretation_t.E5M2
            )
            out_u8[1] = float32_to_fp8(
                x32[0], saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )
            out_u8[2] = float64_to_fp8(
                x64[0], saturation_t.NOSAT, fp8_interpretation_t.E4M3
            )
            out_u8[3] = float64_to_fp8(
                x64[0], saturation_t.SATFINITE, fp8_interpretation_t.E4M3
            )

        out_u8 = np.zeros(4, dtype=np.uint8)
        x32 = np.array([1e20], dtype=np.float32)
        x64 = np.array([1e300], dtype=np.float64)
        kernel[1, 1](out_u8, x32, x64)

        self.assertEqual(out_u8[0], 0x7C)  # E5M2 overflow -> Inf (NOSAT)
        self.assertEqual(
            out_u8[1], 0x7B
        )  # E5M2 overflow -> MAXNORM (SATFINITE)
        self.assertEqual(out_u8[2], 0x7F)  # E4M3 overflow -> NaN (NOSAT)
        self.assertEqual(
            out_u8[3], 0x7E
        )  # E4M3 overflow -> MAXNORM (SATFINITE)

    def test_e8m0_aliases_and_raw_roundtrip(self):
        @cuda.jit
        def kernel(out_u8, out_raw_u16, x32, x64):
            out_u8[0] = float32_to_e8m0(
                x32[0], saturation_t.NOSAT, cudaRoundMode.cudaRoundPosInf
            )
            out_u8[1] = float32_to_e8m0(
                x32[0], saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )
            out_u8[2] = float64_to_e8m0(
                x64[0], saturation_t.NOSAT, cudaRoundMode.cudaRoundPosInf
            )
            out_u8[3] = float64_to_e8m0(
                x64[0], saturation_t.SATFINITE, cudaRoundMode.cudaRoundPosInf
            )

            raw_one = e8m0_to_bfloat16_raw(uint8(127))
            out_raw_u16[0] = raw_one.x

            raw_half = e8m0_to_bfloat16_raw(uint8(126))
            out_u8[4] = bfloat16_raw_to_e8m0(
                raw_half, saturation_t.NOSAT, cudaRoundMode.cudaRoundZero
            )

        x32_over = np.nextafter(np.float32(2.0**127), np.float32(np.inf))
        x64_over = np.nextafter(np.float64(2.0**127), np.float64(np.inf))
        x32 = np.array([x32_over], dtype=np.float32)
        x64 = np.array([x64_over], dtype=np.float64)
        out_u8 = np.zeros(5, dtype=np.uint8)
        out_raw_u16 = np.zeros(1, dtype=np.uint16)
        kernel[1, 1](out_u8, out_raw_u16, x32, x64)

        self.assertEqual(out_u8[0], 0xFF)  # NOSAT overflow -> NaN
        self.assertEqual(out_u8[1], 0xFE)  # SATFINITE overflow -> max finite
        self.assertEqual(out_u8[2], 0xFF)  # NOSAT overflow -> NaN
        self.assertEqual(out_u8[3], 0xFE)  # SATFINITE overflow -> max finite
        self.assertEqual(out_raw_u16[0], 0x3F80)  # 1.0 in BF16 raw format
        self.assertEqual(out_u8[4], 126)  # Roundtrip e8m0 -> bf16 raw -> e8m0

    def test_bfloat16_raw_to_fp8_alias(self):
        @cuda.jit
        def kernel(out_u8):
            # Exponent 254 corresponds to 2^127 in E8M0, out-of-range for FP8 E5M2.
            raw = e8m0_to_bfloat16_raw(uint8(254))
            out_u8[0] = bfloat16_raw_to_fp8(
                raw, saturation_t.NOSAT, fp8_interpretation_t.E5M2
            )
            out_u8[1] = bfloat16_raw_to_fp8(
                raw, saturation_t.SATFINITE, fp8_interpretation_t.E5M2
            )

        out_u8 = np.zeros(2, dtype=np.uint8)
        kernel[1, 1](out_u8)

        self.assertEqual(out_u8[0], 0x7C)  # E5M2 overflow -> Inf (NOSAT)
        self.assertEqual(
            out_u8[1], 0x7B
        )  # E5M2 overflow -> MAXNORM (SATFINITE)


if __name__ == "__main__":
    unittest.main()
