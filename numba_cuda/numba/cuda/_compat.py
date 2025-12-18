# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from packaging import version
from cuda import core


CUDA_CORE_VERSION = version.parse(core.__version__)
if CUDA_CORE_VERSION < version.parse("0.5.0"):
    pass
else:
    pass
