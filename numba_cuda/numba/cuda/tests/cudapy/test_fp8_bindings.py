# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Comprehensive test suite for CUDA FP8 types and conversion functions.

Tests cover:
- fp8_e5m2, fp8_e4m3, fp8_e8m0 types
- Constructors from various numeric types
"""

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

if not config.ENABLE_CUDASIM:
    from numba.cuda._internal.cuda_fp8 import (
        fp8_e5m2,
        fp8_e4m3,
        fp8_e8m0,
    )


class FP8BasicTest(CUDATestCase):
    """Basic constructor and cast tests for FP8 types."""

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


if __name__ == "__main__":
    unittest.main()
