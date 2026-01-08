# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# pyright: reportMissingTypeStubs=false, reportMissingImports=false
# mypy: ignore-errors

import numpy as np
import pytest

from numba import cuda
from numba.cuda import config
from numba.cuda import float32, int8, int16, int32, int64, uint8


def _full_array_str(a):
    # Ensure CI logs show the entire array (no truncation).
    return np.array2string(np.asarray(a), threshold=np.inf)


@pytest.mark.skipif(
    config.ENABLE_CUDASIM, reason="Requires real CUDA hardware (no cudasim)"
)
def test_fp8_e4m3_neg6_to_int8_debug():
    if not cuda.is_available():
        pytest.skip("CUDA is not available")

    from numba.cuda._internal.cuda_fp8 import fp8_e4m3

    dev = cuda.get_current_device()
    print(
        "fp8_e4m3 int8 debug on device:",
        dev.name,
        "cc=",
        dev.compute_capability,
    )

    # out_i64 layout:
    #  0: int8(fp)
    #  1: int8(fp) again (repeat to spot instability)
    #  2: int8(float32(fp)) (control path)
    #  3: int16(fp)
    #  4: int32(fp)
    #  5: int64(fp)
    #  6: uint8(fp)  (numeric conversion; negatives clamp to 0 per CUDA header)
    out_i64 = np.zeros(7, dtype=np.int64)
    out_f32 = np.zeros(2, dtype=np.float32)

    @cuda.jit
    def kernel(out_i64, out_f32):
        if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
            v = float32(-6.0)
            fp = fp8_e4m3(v)

            f = float32(fp)
            out_f32[0] = v
            out_f32[1] = f

            i8a = int8(fp)
            i8b = int8(fp)
            i8_from_f = int8(f)

            out_i64[0] = int64(i8a)
            out_i64[1] = int64(i8b)
            out_i64[2] = int64(i8_from_f)
            out_i64[3] = int64(int16(fp))
            out_i64[4] = int64(int32(fp))
            out_i64[5] = int64(fp)
            out_i64[6] = int64(uint8(fp))

            # Device-side debug print (shows up with pytest -s)
            print(
                "fp8_e4m3(-6) debug:",
                "v=",
                v,
                "float(fp)=",
                f,
                "int8(fp)=",
                int32(i8a),
                "int8(float(fp))=",
                int32(i8_from_f),
                "int16(fp)=",
                int32(int16(fp)),
                "int32(fp)=",
                int32(int32(fp)),
                "int64(fp)=",
                int64(fp),
                "uint8(fp)=",
                int32(uint8(fp)),
            )

    kernel[1, 1](out_i64, out_f32)

    print("host out_f32:", out_f32)
    print("host out_i64:", out_i64)

    # Strong assertions with full array dump on failure (for CI log debugging).
    expected_f32 = np.array([-6.0, -6.0], dtype=np.float32)
    expected_i64 = np.array([-6, -6, -6, -6, -6, -6, 0], dtype=np.int64)

    f32_ok = np.array_equal(out_f32, expected_f32)
    i64_ok = np.array_equal(out_i64, expected_i64)
    if not (f32_ok and i64_ok):
        raise AssertionError(
            "fp8_e4m3(-6.0) conversion mismatch\n"
            f"out_f32={_full_array_str(out_f32)} expected_f32={_full_array_str(expected_f32)}\n"
            f"out_i64={_full_array_str(out_i64)} expected_i64={_full_array_str(expected_i64)}\n"
            "note: out_i64 layout is [int8(fp), int8(fp), int8(float(fp)), int16(fp), int32(fp), int64(fp), uint8(fp)]"
        )
