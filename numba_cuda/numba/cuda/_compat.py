# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from packaging import version
from cuda import core


CUDA_CORE_VERSION = version.parse(core.__version__)
CUDA_CORE_GT_0_6 = CUDA_CORE_VERSION >= version.parse("0.6.0")


def cuda_core_attr_value(obj, name):
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def make_cuda_core_launch_config(**kwargs):
    try:
        return core.LaunchConfig(**kwargs)
    except TypeError as e:
        if "is_cooperative" not in kwargs or "is_cooperative" not in str(e):
            raise

        compat_kwargs = kwargs.copy()
        compat_kwargs["cooperative_launch"] = compat_kwargs.pop(
            "is_cooperative"
        )
        return core.LaunchConfig(**compat_kwargs)
