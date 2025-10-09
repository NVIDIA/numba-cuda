# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
NumPy extensions.
"""

from numba.cuda.np.arraymath import cross2d


__all__ = ["cross2d"]
