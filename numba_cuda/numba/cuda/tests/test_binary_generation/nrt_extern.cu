#include <nrt.cuh>

extern "C" __device__ int device_allocate_deallocate(int* nb_retval){
    auto ptr = NRT_Allocate(1);
    NRT_Free(ptr);
    return 0;
}
