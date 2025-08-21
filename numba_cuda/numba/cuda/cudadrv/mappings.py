# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import config
from . import enums

if config.CUDA_USE_NVIDIA_BINDING:
    from cuda.bindings.driver import CUjitInputType

    FILE_EXTENSION_MAP = {
        "o": CUjitInputType.CU_JIT_INPUT_OBJECT,
        "ptx": CUjitInputType.CU_JIT_INPUT_PTX,
        "a": CUjitInputType.CU_JIT_INPUT_LIBRARY,
        "lib": CUjitInputType.CU_JIT_INPUT_LIBRARY,
        "cubin": CUjitInputType.CU_JIT_INPUT_CUBIN,
        "fatbin": CUjitInputType.CU_JIT_INPUT_FATBINARY,
        "ltoir": CUjitInputType.CU_JIT_INPUT_NVVM,
    }
else:
    FILE_EXTENSION_MAP = {
        "o": enums.CU_JIT_INPUT_OBJECT,
        "ptx": enums.CU_JIT_INPUT_PTX,
        "a": enums.CU_JIT_INPUT_LIBRARY,
        "lib": enums.CU_JIT_INPUT_LIBRARY,
        "cubin": enums.CU_JIT_INPUT_CUBIN,
        "fatbin": enums.CU_JIT_INPUT_FATBINARY,
        "ltoir": enums.CU_JIT_INPUT_NVVM,
    }
