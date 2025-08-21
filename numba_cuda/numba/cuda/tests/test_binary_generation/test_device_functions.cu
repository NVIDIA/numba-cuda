/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cuda_fp16.h>

extern __device__ bool __heq(__half arg1, __half arg2);

__device__ __half test_add_fp16(__half arg1, __half arg2) {
  return __hadd(arg1, arg2);
}

__device__ bool test_cmp_fp16(__half arg1, __half arg2) {
  return __heq(arg1, arg2);
}

typedef unsigned int uint32_t;

extern "C" __device__ int add_from_numba(uint32_t *result, uint32_t a,
                                         uint32_t b) {
  *result = a + b;
  return 0;
}

extern "C" __device__ uint32_t add_cabi(uint32_t a, uint32_t b) {
  return a + b;
}
