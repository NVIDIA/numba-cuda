#ifndef _NRT_H
#define _NRT_H

#include <cuda/atomic>

typedef void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);

typedef struct MemInfo NRT_MemInfo;

extern "C" {
struct MemInfo {
  cuda::atomic<size_t, cuda::thread_scope_device> refct;
  NRT_dtor_function dtor;
  void* dtor_info;
  void* data;
  size_t size;
};
}

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

static __device__ void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align, NRT_MemInfo **mi);
static __device__ void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out);
extern "C" __device__ void* NRT_Allocate_External(size_t size);

/* The Memory System object */
__device__ NRT_MemSys* TheMSys;

extern "C" __device__ void* NRT_Allocate(size_t size)
{
  void* ptr = NULL;
  ptr       = malloc(size);
//  if (TheMSys->stats.enabled) { TheMSys->stats.alloc++; }
  return ptr;
}

extern "C" __device__ void NRT_MemInfo_init(NRT_MemInfo* mi,
                                            void* data,
                                            size_t size,
                                            NRT_dtor_function dtor,
                                            void* dtor_info)
//                                            NRT_MemSys* TheMSys)
{
  mi->refct     = 1; /* starts with 1 refct */
  mi->dtor      = dtor;
  mi->dtor_info = dtor_info;
  mi->data      = data;
  mi->size      = size;
//  if (TheMSys->stats.enabled) { TheMSys->stats.mi_alloc++; }
}

extern "C"
__device__ NRT_MemInfo* NRT_MemInfo_new(
  void* data, size_t size, NRT_dtor_function dtor, void* dtor_info)
{
  NRT_MemInfo* mi = (NRT_MemInfo*)NRT_Allocate(sizeof(NRT_MemInfo));
  if (mi != NULL) { NRT_MemInfo_init(mi, data, size, dtor, dtor_info); }
  return mi;
}

extern "C" __device__ void NRT_Free(void* ptr)
{
  free(ptr);
  //if (TheMSys->stats.enabled) { TheMSys->stats.free++; }
}

extern "C" __device__ void NRT_dealloc(NRT_MemInfo* mi)
{
  NRT_Free(mi);
}

extern "C" __device__ void NRT_MemInfo_destroy(NRT_MemInfo* mi)
{
  NRT_dealloc(mi);
  //if (TheMSys->stats.enabled) { TheMSys->stats.mi_free++; }
}
extern "C" __device__ void NRT_MemInfo_call_dtor(NRT_MemInfo* mi)
{
  if (mi->dtor) /* We have a destructor */
    mi->dtor(mi->data, mi->size, NULL);
  /* Clear and release MemInfo */
  NRT_MemInfo_destroy(mi);
}

extern "C" __device__ void* NRT_MemInfo_data_fast(NRT_MemInfo *mi)
{
  return mi->data;
}

extern "C" __device__ NRT_MemInfo *NRT_MemInfo_alloc_aligned(size_t size, unsigned align) {
    NRT_MemInfo *mi = NULL;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    if (data == NULL) {
        return NULL; /* return early as allocation failed */
    }
    //NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_aligned %p\n", data));
    NRT_MemInfo_init(mi, data, size, NULL, NULL);
    return mi;
}

static
__device__ void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align,
                                          NRT_MemInfo **mi)
{
    size_t offset = 0, intptr = 0, remainder = 0;
    //NRT_Debug(nrt_debug_print("nrt_allocate_meminfo_and_data_align %p\n", allocator));
    char *base = (char *)nrt_allocate_meminfo_and_data(size + 2 * align, mi);
    if (base == NULL) {
        return NULL; /* return early as allocation failed */
    }
    intptr = (size_t) base;
    /*
     * See if the allocation is aligned already...
     * Check if align is a power of 2, if so the modulo can be avoided.
     */
    if((align & (align - 1)) == 0)
    {
        remainder = intptr & (align - 1);
    }
    else
    {
        remainder = intptr % align;
    }
    if (remainder == 0){ /* Yes */
        offset = 0;
    } else { /* No, move forward `offset` bytes */
        offset = align - remainder;
    }
    return (void*)((char *)base + offset);
}

static
__device__ void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out) {
    NRT_MemInfo *mi = NULL;
    //NRT_Debug(nrt_debug_print("nrt_allocate_meminfo_and_data %p\n", allocator));
    char *base = (char *)NRT_Allocate_External(sizeof(NRT_MemInfo) + size);
    if (base == NULL) {
        *mi_out = NULL; /* set meminfo to NULL as allocation failed */
        return NULL; /* return early as allocation failed */
    }
    mi = (NRT_MemInfo *) base;
    *mi_out = mi;
    return (void*)((char *)base + sizeof(NRT_MemInfo));
}

extern "C" __device__ void* NRT_Allocate_External(size_t size) {
    void *ptr = NULL;
    ptr = malloc(size);
    //NRT_Debug(nrt_debug_print("NRT_Allocate_External bytes=%zu ptr=%p\n", size, ptr));

    //if (TheMSys.stats.enabled)
    //{
    //    TheMSys.stats.alloc++;
    //}
    return ptr;
}


/*
  c++ version of the NRT_decref function that usually is added to
  the final kernel link in PTX form by numba. This version may be
  used by c++ APIs that accept ownership of live objects and must
  manage them going forward.
*/
extern "C" __device__ void NRT_decref(NRT_MemInfo* mi)
{
  if (mi != NULL) {
    mi->refct--;
    if (mi->refct == 0) { NRT_MemInfo_call_dtor(mi); }
  }
}

#endif

extern "C" __device__ void NRT_incref(NRT_MemInfo* mi)
{
  if (mi != NULL) {
    mi->refct++;
  }
}
