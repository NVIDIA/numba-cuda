#include <nrt.cuh>

extern "C" __device__ int extern_func(int* nb_retval){
    auto ptr = NRT_Allocate(1);
    NRT_Free(ptr);
    return 0;
}
