..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


Writing Device Functions
========================

CUDA device functions can only be invoked from within the device (by a kernel
or another device function).  To define a device function::

    from numba import cuda

    @cuda.jit(device=True)
    def a_device_function(a, b):
        return a + b

Unlike a kernel function, a device function can return a value like normal
functions.
