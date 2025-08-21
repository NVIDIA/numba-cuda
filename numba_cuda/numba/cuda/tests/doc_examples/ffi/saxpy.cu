/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <add.cuh> // In numba/cuda/tests/data/include
#include <mul.cuh> // In numba/cuda/tests/doc_examples/ffi/include

extern "C"
__device__ int saxpy(float *ret, float a, float x, float y)
{
    *ret = myadd(mymul(a, x), y);
    return 0;
}
