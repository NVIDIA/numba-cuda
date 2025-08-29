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


class DefaultPassBuilder(object):
    @staticmethod
    def define_nopython_lowering_pipeline(state, name="nopython_lowering"):
        pass

    @staticmethod
    def define_typed_pipeline(state, name="typed"):
        pass


class CompilerBase:
    def __init__(
        self, typingctx, targetctx, library, args, return_type, flags, locals
    ):
        pass


PassManager = None
