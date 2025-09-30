# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda


@cuda.jit(device=True)
def cuda_module_in_device_function():
    return cuda.threadIdx.x
