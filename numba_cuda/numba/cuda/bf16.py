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
    # Comparison intrinsics
    __heq as heq,
    __hne as hne,
    __hge as hge,
    __hgt as hgt,
    __hle as hle,
    __hlt as hlt,
    __hmax as hmax,
    __hmin as hmin,
    __hmax_nan as hmax_nan,
    __hmin_nan as hmin_nan,
    __hisinf as hisinf,
    __hisnan as hisnan,
    # Unordered comparison intrinsics
    __hequ as hequ,
    __hneu as hneu,
    __hgeu as hgeu,
    __hgtu as hgtu,
    __hleu as hleu,
    __hltu as hltu,
    # Precision conversion and data movement
    __bfloat162float as bfloat162float,
    __float2bfloat16_rn as float2bfloat16_rn,
    __float2bfloat16_rz as float2bfloat16_rz,
    __float2bfloat16_rd as float2bfloat16_rd,
    __float2bfloat16_ru as float2bfloat16_ru,
    __int2bfloat16_rn as int2bfloat16_rn,
    __int2bfloat16_rz as int2bfloat16_rz,
    __int2bfloat16_rd as int2bfloat16_rd,
    __int2bfloat16_ru as int2bfloat16_ru,
    __bfloat162int_rn as bfloat162int_rn,
    __bfloat162int_rz as bfloat162int_rz,
    __bfloat162int_rd as bfloat162int_rd,
    __bfloat162int_ru as bfloat162int_ru,
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
    # Comparison intrinsics
    "heq",
    "hne",
    "hge",
    "hgt",
    "hle",
    "hlt",
    "hmax",
    "hmin",
    "hmax_nan",
    "hmin_nan",
    "hisinf",
    "hisnan",
    "hequ",
    "hneu",
    "hgeu",
    "hgtu",
    "hleu",
    "hltu",
    # Precision conversion and data movement
    "bfloat162float",
    "float2bfloat16_rn",
    "float2bfloat16_rz",
    "float2bfloat16_rd",
    "float2bfloat16_ru",
    "int2bfloat16_rn",
    "int2bfloat16_rz",
    "int2bfloat16_rd",
    "int2bfloat16_ru",
    "bfloat162int_rn",
    "bfloat162int_rz",
    "bfloat162int_rd",
    "bfloat162int_ru",
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
