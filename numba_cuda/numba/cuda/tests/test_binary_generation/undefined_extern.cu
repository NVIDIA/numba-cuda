extern __device__ float undef(float a, float b);

__global__ void f(float *r, float *a, float *b) { r[0] = undef(a[0], b[0]); }
