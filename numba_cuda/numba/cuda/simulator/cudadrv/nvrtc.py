# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
NVVM is not supported in the simulator, but stubs are provided to allow tests
to import correctly.
"""


def compile(src, name, cc, ltoir=False, lineinfo=False, debug=False):
    raise RuntimeError("NVRTC is not supported in the simulator")
