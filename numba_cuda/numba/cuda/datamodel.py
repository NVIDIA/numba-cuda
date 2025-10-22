# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


import importlib
import sys
from numba.cuda.utils import _RedirectSubpackage

if importlib.util.find_spec("numba.core.datamodel"):
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.core.datamodel"
    )
else:
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.cuda.cuda_datamodel"
    )
