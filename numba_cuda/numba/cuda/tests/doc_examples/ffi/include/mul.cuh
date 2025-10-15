/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

// Templated multiplication function: mymul
template <typename T>
__device__ T mymul(T a, T b) { return a * b; }
