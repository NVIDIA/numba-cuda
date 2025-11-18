/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cuda/atomic>

// Globally needed variables
struct NRT_MemSys {
    struct {
      bool enabled;
      cuda::atomic<size_t, cuda::thread_scope_device> alloc;
      cuda::atomic<size_t, cuda::thread_scope_device> free;
      cuda::atomic<size_t, cuda::thread_scope_device> mi_alloc;
      cuda::atomic<size_t, cuda::thread_scope_device> mi_free;
    } stats;
  };

/* The Memory System object */
__device__ NRT_MemSys* TheMSys;

extern "C" __global__ void NRT_MemSys_set(NRT_MemSys *memsys_ptr);
