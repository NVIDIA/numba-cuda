# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.core.targetconfig import TargetConfig, Option

from numba.cuda.core.options import (
    ParallelOptions,
    FastMathOptions,
    InlineOptions,
)

from numba.cuda.core.callconv import BaseCallConv


class Flags(TargetConfig):
    __slots__ = ()

    enable_looplift = Option(
        type=bool,
        default=False,
        doc="Enable loop-lifting",
    )
    enable_pyobject = Option(
        type=bool,
        default=False,
        doc="Enable pyobject mode (in general)",
    )
    enable_pyobject_looplift = Option(
        type=bool,
        default=False,
        doc="Enable pyobject mode inside lifted loops",
    )
    enable_ssa = Option(
        type=bool,
        default=True,
        doc="Enable SSA",
    )
    force_pyobject = Option(
        type=bool,
        default=False,
        doc="Force pyobject mode inside the whole function",
    )
    release_gil = Option(
        type=bool,
        default=False,
        doc="Release GIL inside the native function",
    )
    no_compile = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    debuginfo = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    boundscheck = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    forceinline = Option(
        type=bool,
        default=False,
        doc="Force inlining of the function. Overrides _dbg_optnone.",
    )
    no_cpython_wrapper = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    no_cfunc_wrapper = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    auto_parallel = Option(
        type=ParallelOptions,
        default=ParallelOptions(False),
        doc="""Enable automatic parallel optimization, can be fine-tuned by
taking a dictionary of sub-options instead of a boolean, see parfor.py for
detail""",
    )
    nrt = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    no_rewrites = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    error_model = Option(
        type=str,
        default="python",
        doc="TODO",
    )
    fastmath = Option(
        type=FastMathOptions,
        default=FastMathOptions(False),
        doc="TODO",
    )
    noalias = Option(
        type=bool,
        default=False,
        doc="TODO",
    )
    inline = Option(
        type=InlineOptions,
        default=InlineOptions("never"),
        doc="TODO",
    )

    dbg_extend_lifetimes = Option(
        type=bool,
        default=False,
        doc=(
            "Extend variable lifetime for debugging. "
            "This automatically turns on with debug=True."
        ),
    )

    dbg_optnone = Option(
        type=bool,
        default=False,
        doc=(
            "Disable optimization for debug. "
            "Equivalent to adding optnone attribute in the LLVM Function."
        ),
    )

    dbg_directives_only = Option(
        type=bool,
        default=False,
        doc=(
            "Make debug emissions directives-only. "
            "Used when generating lineinfo."
        ),
    )


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.nrt = True


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


def _call_conv_options_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, BaseCallConv)
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

    call_conv = Option(type=_call_conv_options_type, default=None, doc="")
