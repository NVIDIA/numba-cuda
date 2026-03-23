# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from numba.cuda._internal.cuda_fp8 import (
    typing_registry,
    target_registry,
)


__all__ = [
    "typing_registry",
    "target_registry",
]
