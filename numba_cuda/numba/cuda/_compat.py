# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from packaging import version
from cuda import core


CUDA_CORE_VERSION = version.parse(core.__version__)
CUDA_CORE_GT_0_6 = CUDA_CORE_VERSION >= version.parse("0.6.0")
CUDA_CORE_GE_1_0 = CUDA_CORE_VERSION >= version.parse("1.0.0")
