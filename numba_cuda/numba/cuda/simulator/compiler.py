# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
The compiler is not implemented in the simulator. This module provides a stub
to allow tests to import successfully.
"""

compile = None
compile_for_current_device = None
compile_ptx = None
compile_ptx_for_current_device = None
declare_device_function = None


def run_frontend(func):
    pass