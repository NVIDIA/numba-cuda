#include <nrt.cuh>

extern "C" __device__ int extern_func(int* nb_retval){
    NRT_Allocate(1);
    return 0;
}
