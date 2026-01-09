# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
from numba.cuda.utils import redirect_numba_module

sys.modules[__name__] = redirect_numba_module(
    locals(), "numba.core.errors", "numba.cuda.core.cuda_errors"
)
