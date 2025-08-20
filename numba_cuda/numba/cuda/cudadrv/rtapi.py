# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Declarations of the Runtime API functions.
"""

from ctypes import c_int, POINTER

API_PROTOTYPES = {
    # cudaError_t cudaRuntimeGetVersion ( int* runtimeVersion )
    "cudaRuntimeGetVersion": (c_int, POINTER(c_int)),
}
