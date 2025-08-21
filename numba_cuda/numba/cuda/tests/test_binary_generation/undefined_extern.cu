/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

extern __device__ float undef(float a, float b);

__global__ void f(float *r, float *a, float *b) { r[0] = undef(a[0], b[0]); }
