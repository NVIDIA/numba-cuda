# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from numba.cuda._internal.cuda_fp8 import (
    typing_registry,
    target_registry,
    saturation_t,
    fp8_interpretation_t,
    fp8_e5m2,
    fp8x2_e5m2,
    fp8x4_e5m2,
    fp8_e4m3,
    fp8x2_e4m3,
    fp8x4_e4m3,
    fp8_e8m0,
    fp8x2_e8m0,
    fp8x4_e8m0,
    cvt_double_to_fp8,
    cvt_double2_to_fp8x2,
    cvt_float_to_fp8,
    cvt_float2_to_fp8x2,
    cvt_bfloat16raw_to_fp8,
    cvt_bfloat16raw_to_e8m0,
    cvt_float_to_e8m0,
    cvt_float2_to_e8m0x2,
    cvt_double_to_e8m0,
    cvt_double2_to_e8m0x2,
    cvt_e8m0_to_bf16raw,
)


# Public aliases using Numba/Numpy-style conversion names.
SaturationMode = saturation_t
FP8Format = fp8_interpretation_t
float32_to_fp8 = cvt_float_to_fp8
float64_to_fp8 = cvt_double_to_fp8
float32x2_to_fp8x2 = cvt_float2_to_fp8x2
float64x2_to_fp8x2 = cvt_double2_to_fp8x2
bfloat16_raw_to_fp8 = cvt_bfloat16raw_to_fp8
bfloat16_raw_to_e8m0 = cvt_bfloat16raw_to_e8m0
float32_to_e8m0 = cvt_float_to_e8m0
float64_to_e8m0 = cvt_double_to_e8m0
float32x2_to_e8m0x2 = cvt_float2_to_e8m0x2
float64x2_to_e8m0x2 = cvt_double2_to_e8m0x2
e8m0_to_bfloat16_raw = cvt_e8m0_to_bf16raw


__all__ = [
    "typing_registry",
    "target_registry",
    "SaturationMode",
    "FP8Format",
    "saturation_t",
    "fp8_interpretation_t",
    "fp8_e5m2",
    "fp8x2_e5m2",
    "fp8x4_e5m2",
    "fp8_e4m3",
    "fp8x2_e4m3",
    "fp8x4_e4m3",
    "fp8_e8m0",
    "fp8x2_e8m0",
    "fp8x4_e8m0",
    "cvt_double_to_fp8",
    "cvt_double2_to_fp8x2",
    "cvt_float_to_fp8",
    "cvt_float2_to_fp8x2",
    "cvt_bfloat16raw_to_fp8",
    "cvt_bfloat16raw_to_e8m0",
    "cvt_float_to_e8m0",
    "cvt_float2_to_e8m0x2",
    "cvt_double_to_e8m0",
    "cvt_double2_to_e8m0x2",
    "cvt_e8m0_to_bf16raw",
    "float32_to_fp8",
    "float64_to_fp8",
    "float32x2_to_fp8x2",
    "float64x2_to_fp8x2",
    "bfloat16_raw_to_fp8",
    "bfloat16_raw_to_e8m0",
    "float32_to_e8m0",
    "float64_to_e8m0",
    "float32x2_to_e8m0x2",
    "float64x2_to_e8m0x2",
    "e8m0_to_bfloat16_raw",
]
