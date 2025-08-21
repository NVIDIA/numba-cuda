# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba.core.types as types
from numba.cuda._internal.cuda_fp16 import (
    typing_registry,
    target_registry,
    __half,
    __double2half,
    __float2half,
    __float2half_rd,
    __float2half_rn,
    __float2half_ru,
    __float2half_rz,
    __int2half_rd,
    __int2half_rn,
    __int2half_ru,
    __int2half_rz,
    __ll2half_rd,
    __ll2half_rn,
    __ll2half_ru,
    __ll2half_rz,
    __short2half_rd,
    __short2half_rn,
    __short2half_ru,
    __short2half_rz,
    __uint2half_rd,
    __uint2half_rn,
    __uint2half_ru,
    __uint2half_rz,
    __ull2half_rd,
    __ull2half_rn,
    __ull2half_ru,
    __ull2half_rz,
    __ushort2half_rd,
    __ushort2half_rn,
    __ushort2half_ru,
    __ushort2half_rz,
    __half2char_rz,
    __half2float,
    __half2int_rd,
    __half2int_rn,
    __half2int_ru,
    __half2int_rz,
    __half2ll_rd,
    __half2ll_rn,
    __half2ll_ru,
    __half2ll_rz,
    __half2short_rd,
    __half2short_rn,
    __half2short_ru,
    __half2short_rz,
    __half2uchar_rz,
    __half2uint_rd,
    __half2uint_rn,
    __half2uint_ru,
    __half2uint_rz,
    __half2ull_rd,
    __half2ull_rn,
    __half2ull_ru,
    __half2ull_rz,
    __half2ushort_rd,
    __half2ushort_rn,
    __half2ushort_ru,
    __half2ushort_rz,
    __short_as_half,
    __ushort_as_half,
    __half_as_short,
    __half_as_ushort,
    __habs as habs,
    __habs,
    __hadd as hadd,
    __hadd,
    __hadd_rn,
    __hadd_sat,
    __hcmadd,
    __hdiv as hdiv,
    __hdiv,
    __heq as heq,
    __heq,
    __hequ,
    __hfma as hfma,
    __hfma,
    __hfma_relu,
    __hfma_sat,
    __hge as hge,
    __hge,
    __hgeu,
    __hgt as hgt,
    __hgt,
    __hgtu,
    __hisinf,
    __hisnan,
    __hle as hle,
    __hle,
    __hleu,
    __hlt as hlt,
    __hlt,
    __hltu,
    __hmax as hmax,
    __hmax,
    __hmax_nan,
    __hmin as hmin,
    __hmin,
    __hmin_nan,
    __hmul as hmul,
    __hmul,
    __hmul_rn,
    __hmul_sat,
    __hne as hne,
    __hne,
    __hneg as hneg,
    __hneg,
    __hneu,
    __hsub as hsub,
    __hsub,
    __hsub_rn,
    __hsub_sat,
    atomicAdd,
    hceil,
    hcos,
    hexp,
    hexp10,
    hexp2,
    hfloor,
    hlog,
    hlog10,
    hlog2,
    hrcp,
    hrint,
    hrsqrt,
    hsin,
    hsqrt,
    htanh,
    htanh_approx,
    htrunc,
)

from numba.extending import overload
import math


def _make_unary(a, func):
    if isinstance(a, types.Float) and a.bitwidth == 16:
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


@overload(math.fabs, target="cuda")
def fabs_ol(a):
    return _make_unary(a, habs)


@overload(math.sqrt, target="cuda")
def sqrt_ol(a):
    return _make_unary(a, hsqrt)


@overload(math.log, target="cuda")
def log_ol(a):
    return _make_unary(a, hlog)


@overload(math.log2, target="cuda")
def log2_ol(a):
    return _make_unary(a, hlog2)


@overload(math.log10, target="cuda")
def log10_ol(a):
    return _make_unary(a, hlog10)


@overload(math.exp, target="cuda")
def exp_ol(a):
    return _make_unary(a, hexp)


@overload(math.tanh, target="cuda")
def tanh_ol(a):
    return _make_unary(a, htanh)


@overload(math.cos, target="cuda")
def cos_ol(a):
    return _make_unary(a, hcos)


@overload(math.sin, target="cuda")
def sin_ol(a):
    return _make_unary(a, hsin)


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
    "__half",
    "__double2half",
    "__float2half",
    "__float2half_rd",
    "__float2half_rn",
    "__float2half_ru",
    "__float2half_rz",
    "__int2half_rd",
    "__int2half_rn",
    "__int2half_ru",
    "__int2half_rz",
    "__ll2half_rd",
    "__ll2half_rn",
    "__ll2half_ru",
    "__ll2half_rz",
    "__short2half_rd",
    "__short2half_rn",
    "__short2half_ru",
    "__short2half_rz",
    "__uint2half_rd",
    "__uint2half_rn",
    "__uint2half_ru",
    "__uint2half_rz",
    "__ull2half_rd",
    "__ull2half_rn",
    "__ull2half_ru",
    "__ull2half_rz",
    "__ushort2half_rd",
    "__ushort2half_rn",
    "__ushort2half_ru",
    "__ushort2half_rz",
    "__half2char_rz",
    "__half2float",
    "__half2int_rd",
    "__half2int_rn",
    "__half2int_ru",
    "__half2int_rz",
    "__half2ll_rd",
    "__half2ll_rn",
    "__half2ll_ru",
    "__half2ll_rz",
    "__half2short_rd",
    "__half2short_rn",
    "__half2short_ru",
    "__half2short_rz",
    "__half2uchar_rz",
    "__half2uint_rd",
    "__half2uint_rn",
    "__half2uint_ru",
    "__half2uint_rz",
    "__half2ull_rd",
    "__half2ull_rn",
    "__half2ull_ru",
    "__half2ull_rz",
    "__half2ushort_rd",
    "__half2ushort_rn",
    "__half2ushort_ru",
    "__half2ushort_rz",
    "__short_as_half",
    "__ushort_as_half",
    "__half_as_short",
    "__half_as_ushort",
    "habs",
    "__habs",
    "hadd",
    "__hadd",
    "__hadd_rn",
    "__hadd_sat",
    "__hcmadd",
    "hdiv",
    "__hdiv",
    "heq",
    "__heq",
    "__hequ",
    "hfma",
    "__hfma",
    "__hfma_relu",
    "__hfma_sat",
    "hge",
    "__hge",
    "__hgeu",
    "hgt",
    "__hgt",
    "__hgtu",
    "__hisinf",
    "__hisnan",
    "hle",
    "__hle",
    "__hleu",
    "hlt",
    "__hlt",
    "__hltu",
    "hmax",
    "__hmax",
    "__hmax_nan",
    "hmin",
    "__hmin",
    "__hmin_nan",
    "hmul",
    "__hmul",
    "__hmul_rn",
    "__hmul_sat",
    "hne",
    "__hne",
    "hneg",
    "__hneg",
    "__hneu",
    "hsub",
    "__hsub",
    "__hsub_rn",
    "__hsub_sat",
    "atomicAdd",
    "hceil",
    "hcos",
    "hexp",
    "hexp10",
    "hexp2",
    "hfloor",
    "hlog",
    "hlog10",
    "hlog2",
    "hrcp",
    "hrint",
    "hrsqrt",
    "hsin",
    "hsqrt",
    "htanh",
    "htanh_approx",
    "htrunc",
]
