# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from numba.core.compiler import Flags, Option


def _nvvm_options_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, dict)
        return x


def _optional_int_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, int)
        return x


class CUDAFlags(Flags):
    nvvm_options = Option(
        type=_nvvm_options_type,
        default=None,
        doc="NVVM options",
    )
    compute_capability = Option(
        type=tuple,
        default=None,
        doc="Compute Capability",
    )
    max_registers = Option(
        type=_optional_int_type, default=None, doc="Max registers"
    )
    lto = Option(type=bool, default=False, doc="Enable Link-time Optimization")
