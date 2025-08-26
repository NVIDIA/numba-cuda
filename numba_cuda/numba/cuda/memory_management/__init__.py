# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.memory_management.nrt import rtsys  # noqa: F401
from numba.cuda.memory_management import arrayobj_extras  # noqa: F401


__all__ = ["rtsys"]
