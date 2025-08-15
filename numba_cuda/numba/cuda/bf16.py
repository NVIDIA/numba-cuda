from numba.cuda._internal.cuda_bf16 import (
    typing_registry,
    target_registry,
    nv_bfloat16 as bfloat16,
    # Arithmetic intrinsics
    __habs as habs,
    __hadd as hadd,
    __hsub as hsub,
    __hmul as hmul,
    __hadd_rn as hadd_rn,
    __hsub_rn as hsub_rn,
    __hmul_rn as hmul_rn,
    __hdiv as hdiv,
    __hadd_sat as hadd_sat,
    __hsub_sat as hsub_sat,
    __hmul_sat as hmul_sat,
    __hfma as hfma,
    __hfma_sat as hfma_sat,
    __hneg as hneg,
    __hfma_relu as hfma_relu,
    htrunc,
    hceil,
    hfloor,
    hrint,
    hsqrt,
    hrsqrt,
    hrcp,
    hlog,
    hlog2,
    hlog10,
    hcos,
    hsin,
    hexp,
    hexp2,
    hexp10,
    htanh,
    htanh_approx,
)
from numba.extending import overload

import math


def _make_unary(a, func):
    if a == bfloat16:
        return lambda a: func(a)


# Bind low++ bindings to math APIs
@overload(math.trunc, target="cuda")
def trunc_ol(a):
    return _make_unary(a, htrunc)


@overload(math.ceil, target="cuda")
def ceil_ol(a):
    return _make_unary(a, hceil)


@overload(math.floor, target="cuda")
def floor_ol(a):
    return _make_unary(a, hfloor)


@overload(math.sqrt, target="cuda")
def sqrt_ol(a):
    return _make_unary(a, hsqrt)


@overload(math.log, target="cuda")
def log_ol(a):
    return _make_unary(a, hlog)


@overload(math.log10, target="cuda")
def log10_ol(a):
    return _make_unary(a, hlog10)


@overload(math.cos, target="cuda")
def cos_ol(a):
    return _make_unary(a, hcos)


@overload(math.sin, target="cuda")
def sin_ol(a):
    return _make_unary(a, hsin)


@overload(math.tanh, target="cuda")
def tanh_ol(a):
    return _make_unary(a, htanh)


@overload(math.exp, target="cuda")
def exp_ol(a):
    return _make_unary(a, hexp)


try:
    from math import exp2

    @overload(exp2, target="cuda")
    def exp2_ol(a):
        return _make_unary(a, hexp2)
except ImportError:
    pass

__all__ = [
    "typing_registry",
    "target_registry",
    "bfloat16",
    # Arithmetic intrinsics
    "habs",
    "hadd",
    "hsub",
    "hmul",
    "hadd_rn",
    "hsub_rn",
    "hmul_rn",
    "hdiv",
    "hadd_sat",
    "hsub_sat",
    "hmul_sat",
    "hfma",
    "hfma_sat",
    "hneg",
    "hfma_relu",
    "htrunc",
    "hceil",
    "hfloor",
    "hrint",
    "hsqrt",
    "hrsqrt",
    "hrcp",
    "hlog",
    "hlog2",
    "hlog10",
    "hcos",
    "hsin",
    "htanh",
    "htanh_approx",
    "hexp",
    "hexp2",
    "hexp10",
]
