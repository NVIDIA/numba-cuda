/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cuda/atomic>

typedef void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);

extern "C"
struct MemInfo {
  cuda::atomic<size_t, cuda::thread_scope_device> refct;
  NRT_dtor_function dtor;
  void* dtor_info;
  void* data;
  size_t size;
};
typedef struct MemInfo NRT_MemInfo;

extern "C" __device__ void* NRT_Allocate(size_t size);
extern "C" __device__ void NRT_MemInfo_init(NRT_MemInfo* mi,
                                            void* data,
                                            size_t size,
                                            NRT_dtor_function dtor,
                                            void* dtor_info);
static __device__ void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align, NRT_MemInfo **mi);
static __device__ void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out);
extern "C" __device__ void* NRT_Allocate_External(size_t size);
extern "C" __device__ void NRT_decref(NRT_MemInfo* mi);
extern "C" __device__ void NRT_incref(NRT_MemInfo* mi);
extern "C" __device__ void* NRT_Allocate_External(size_t size);
static __device__ void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out);
static __device__ void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align, NRT_MemInfo **mi);
extern "C" __device__ NRT_MemInfo *NRT_MemInfo_alloc_aligned(size_t size, unsigned align);
extern "C" __device__ void* NRT_MemInfo_data_fast(NRT_MemInfo *mi);
extern "C" __device__ void NRT_MemInfo_call_dtor(NRT_MemInfo* mi);
extern "C" __device__ void NRT_MemInfo_destroy(NRT_MemInfo* mi);
extern "C" __device__ void NRT_dealloc(NRT_MemInfo* mi);
extern "C" __device__ void NRT_Free(void* ptr);
extern "C" __device__ NRT_MemInfo* NRT_MemInfo_new(void* data, size_t size, NRT_dtor_function dtor, void* dtor_info);
extern "C" __device__ void NRT_MemInfo_init(NRT_MemInfo* mi,
                                            void* data,
                                            size_t size,
                                            NRT_dtor_function dtor,
                                            void* dtor_info);
