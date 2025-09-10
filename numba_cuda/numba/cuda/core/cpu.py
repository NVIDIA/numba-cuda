# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


from numba.core import (
    types,
    config,
    cgutils,
)
from numba.core.options import TargetOptions, include_default_options

# Re-export these options, they are used from the cpu module throughout the code
# base.
from numba.cuda.core.options import (
    ParallelOptions,  # noqa F401
    FastMathOptions,  # noqa F401
    InlineOptions,  # noqa F401
)  # noqa F401

# Keep those structures in sync with _dynfunc.c.


class ClosureBody(cgutils.Structure):
    _fields = [("env", types.pyobject)]


class EnvBody(cgutils.Structure):
    _fields = [
        ("globals", types.pyobject),
        ("consts", types.pyobject),
    ]


# ----------------------------------------------------------------------------
# TargetOptions

_options_mixin = include_default_options(
    "nopython",
    "forceobj",
    "looplift",
    "_nrt",
    "debug",
    "boundscheck",
    "nogil",
    "no_rewrites",
    "no_cpython_wrapper",
    "no_cfunc_wrapper",
    "parallel",
    "fastmath",
    "error_model",
    "inline",
    "forceinline",
    "_dbg_extend_lifetimes",
    "_dbg_optnone",
)


class CPUTargetOptions(_options_mixin, TargetOptions):
    def finalize(self, flags, options):
        if not flags.is_set("enable_pyobject"):
            flags.enable_pyobject = True

        if not flags.is_set("enable_looplift"):
            flags.enable_looplift = True

        flags.inherit_if_not_set("nrt", default=True)

        if not flags.is_set("debuginfo"):
            flags.debuginfo = config.DEBUGINFO_DEFAULT

        if not flags.is_set("dbg_extend_lifetimes"):
            if flags.debuginfo:
                # auto turn on extend-lifetimes if debuginfo is on and
                # dbg_extend_lifetimes is not set
                flags.dbg_extend_lifetimes = True
            else:
                # set flag using env-var config
                flags.dbg_extend_lifetimes = config.EXTEND_VARIABLE_LIFETIMES

        if not flags.is_set("boundscheck"):
            flags.boundscheck = flags.debuginfo

        flags.enable_pyobject_looplift = True

        flags.inherit_if_not_set("fastmath")

        flags.inherit_if_not_set("error_model", default="python")

        flags.inherit_if_not_set("forceinline")

        if flags.forceinline:
            # forceinline turns off optnone, just like clang.
            flags.dbg_optnone = False
