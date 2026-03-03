# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
NVVM is not supported in the simulator, but stubs are provided to allow tests
to import correctly.
"""


class NvvmSupportError(ImportError):
    pass


class NVVM:
    def __init__(self):
        raise NvvmSupportError("NVVM not supported in the simulator")


CompilationUnit = None
compile_ir = None
set_cuda_kernel = None
get_arch_option = None
LibDevice = None
NvvmError = None


def is_available():
    return False


def get_supported_ccs():
    return ()
