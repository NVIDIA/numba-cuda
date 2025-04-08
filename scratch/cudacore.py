from cuda.core.experimental import Program, ProgramOptions, ObjectCode


SAXPY_KERNEL = """
template<typename T>
__global__ void saxpy(const T a,
                    const T* x,
                    const T* y,
                    T* out,
                    size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"""

prog = Program(SAXPY_KERNEL, code_type="c++")
mod = prog.compile(
    "cubin",
    name_expressions=("saxpy<float>", "saxpy<double>"),
)

cubin = mod._module
sym_map = mod._sym_map

print(sym_map)

mod = ObjectCode.from_cubin(cubin, symbol_mapping=sym_map)
assert mod.code == cubin
mod.get_kernel("saxpy<double>")