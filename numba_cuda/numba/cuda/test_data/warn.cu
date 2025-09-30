/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

extern "C" __device__
int bar(int* out, int a) {
  // Explicitly placed to generate a warning for testing the NVRTC program log
  int unused;
  *out = a * 2;
  return 0;
}
