/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

// Not all CUDA includes are safe to include in device code compiled by NVRTC,
// because it does not have paths to all system include directories. Headers
// such as cuda_device_runtime_api.h are safe to use in NVRTC without adding
// additional includes.
#include <cuda_device_runtime_api.h>
