from numba import config
from cuda import cuda
from . import enums
if config.CUDA_USE_NVIDIA_BINDING:
    jitty = cuda.CUjitInputType
    FILE_EXTENSION_MAP = {
        'o': jitty.CU_JIT_INPUT_OBJECT,
        'ptx': jitty.CU_JIT_INPUT_PTX,
        'a': jitty.CU_JIT_INPUT_LIBRARY,
        'lib': jitty.CU_JIT_INPUT_LIBRARY,
        'cubin': jitty.CU_JIT_INPUT_CUBIN,
        'fatbin': jitty.CU_JIT_INPUT_FATBINARY,
    }
else:
    FILE_EXTENSION_MAP = {
        'o': enums.CU_JIT_INPUT_OBJECT,
        'ptx': enums.CU_JIT_INPUT_PTX,
        'a': enums.CU_JIT_INPUT_LIBRARY,
        'lib': enums.CU_JIT_INPUT_LIBRARY,
        'cubin': enums.CU_JIT_INPUT_CUBIN,
        'fatbin': enums.CU_JIT_INPUT_FATBINARY,
    }
