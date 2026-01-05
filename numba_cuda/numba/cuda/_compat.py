# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from packaging import version
from cuda import core


CUDA_CORE_VERSION = version.parse(core.__version__)
if CUDA_CORE_VERSION < version.parse("0.5.0"):
    from cuda.core.experimental import (
        Program,
        ProgramOptions,
        Linker,
        LinkerOptions,
        Stream,
        Device,
        launch,
        ObjectCode,
        LaunchConfig,
    )
    from cuda.core.experimental._utils.cuda_utils import CUDAError, NVRTCError
else:
    from cuda.core import (
        Program,
        ProgramOptions,
        Linker,
        LinkerOptions,
        Stream,
        Device,
        launch,
        ObjectCode,
        LaunchConfig,
    )
    from cuda.core._utils.cuda_utils import CUDAError, NVRTCError

__all__ = [
    "Program",
    "ProgramOptions",
    "Linker",
    "LinkerOptions",
    "Stream",
    "Device",
    "launch",
    "CUDAError",
    "NVRTCError",
    "ObjectCode",
    "LaunchConfig",
]
