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
extern "C" __device__ void NRT_decref(NRT_MemInfo* mi);

