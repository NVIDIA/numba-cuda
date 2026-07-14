# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import sys

from numba.cuda._internal.cuda_bf16 import (
    typing_registry,
    target_registry,
    nv_bfloat16 as bfloat16,
    _type_unnamed1405307 as _bfloat16_raw_type,
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
    # - floating-point family
    __bfloat162float as bfloat162float,
    __float2bfloat16 as float2bfloat16,
    __double2bfloat16 as double2bfloat16,
    __float2bfloat16_rn as float2bfloat16_rn,
    __float2bfloat16_rz as float2bfloat16_rz,
    __float2bfloat16_rd as float2bfloat16_rd,
    __float2bfloat16_ru as float2bfloat16_ru,
    # - char family
    __bfloat162char_rz as bfloat162char_rz,
    __bfloat162uchar_rz as bfloat162uchar_rz,
    # - int family (signed 32-bit)
    __int2bfloat16_rn as int2bfloat16_rn,
    __int2bfloat16_rz as int2bfloat16_rz,
    __int2bfloat16_rd as int2bfloat16_rd,
    __int2bfloat16_ru as int2bfloat16_ru,
    __bfloat162int_rn as bfloat162int_rn,
    __bfloat162int_rz as bfloat162int_rz,
    __bfloat162int_rd as bfloat162int_rd,
    __bfloat162int_ru as bfloat162int_ru,
    # - short family (signed 16-bit)
    __short2bfloat16_rn as short2bfloat16_rn,
    __short2bfloat16_rz as short2bfloat16_rz,
    __short2bfloat16_rd as short2bfloat16_rd,
    __short2bfloat16_ru as short2bfloat16_ru,
    __bfloat162short_rn as bfloat162short_rn,
    __bfloat162short_rz as bfloat162short_rz,
    __bfloat162short_rd as bfloat162short_rd,
    __bfloat162short_ru as bfloat162short_ru,
    # - ushort family (unsigned 16-bit)
    __ushort2bfloat16_rn as ushort2bfloat16_rn,
    __ushort2bfloat16_rz as ushort2bfloat16_rz,
    __ushort2bfloat16_rd as ushort2bfloat16_rd,
    __ushort2bfloat16_ru as ushort2bfloat16_ru,
    __bfloat162ushort_rn as bfloat162ushort_rn,
    __bfloat162ushort_rz as bfloat162ushort_rz,
    __bfloat162ushort_rd as bfloat162ushort_rd,
    __bfloat162ushort_ru as bfloat162ushort_ru,
    # - uint family (unsigned 32-bit)
    __uint2bfloat16_rn as uint2bfloat16_rn,
    __uint2bfloat16_rz as uint2bfloat16_rz,
    __uint2bfloat16_rd as uint2bfloat16_rd,
    __uint2bfloat16_ru as uint2bfloat16_ru,
    __bfloat162uint_rn as bfloat162uint_rn,
    __bfloat162uint_rz as bfloat162uint_rz,
    __bfloat162uint_rd as bfloat162uint_rd,
    __bfloat162uint_ru as bfloat162uint_ru,
    # - ll family (signed 64-bit)
    __ll2bfloat16_rn as ll2bfloat16_rn,
    __ll2bfloat16_rz as ll2bfloat16_rz,
    __ll2bfloat16_rd as ll2bfloat16_rd,
    __ll2bfloat16_ru as ll2bfloat16_ru,
    __bfloat162ll_rn as bfloat162ll_rn,
    __bfloat162ll_rz as bfloat162ll_rz,
    __bfloat162ll_rd as bfloat162ll_rd,
    __bfloat162ll_ru as bfloat162ll_ru,
    # - ull family (unsigned 64-bit)
    __ull2bfloat16_rn as ull2bfloat16_rn,
    __ull2bfloat16_rz as ull2bfloat16_rz,
    __ull2bfloat16_rd as ull2bfloat16_rd,
    __ull2bfloat16_ru as ull2bfloat16_ru,
    __bfloat162ull_rn as bfloat162ull_rn,
    __bfloat162ull_rz as bfloat162ull_rz,
    __bfloat162ull_rd as bfloat162ull_rd,
    __bfloat162ull_ru as bfloat162ull_ru,
    # - bit reinterpret casts
    __bfloat16_as_short as bfloat16_as_short,
    __bfloat16_as_ushort as bfloat16_as_ushort,
    __short_as_bfloat16 as short_as_bfloat16,
    __ushort_as_bfloat16 as ushort_as_bfloat16,
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
from numba.cuda.extending import intrinsic, overload
from numba.cuda.typing import signature

import math


@intrinsic
def _bfloat16_as_bfloat16_raw(typingctx, value):
    """Cast bfloat16 to bfloat16 raw storage."""
    if value != bfloat16:
        return None

    sig = signature(_bfloat16_raw_type, bfloat16)

    def codegen(context, builder, signature, args):
        return context.cast(
            builder,
            args[0],
            signature.args[0],
            signature.return_type,
        )

    return sig, codegen


@intrinsic
def _bfloat16_raw_as_bfloat16(typingctx, value):
    """Cast bfloat16 raw storage to bfloat16."""
    if value != _bfloat16_raw_type:
        return None

    sig = signature(bfloat16, _bfloat16_raw_type)

    def codegen(context, builder, signature, args):
        return context.cast(
            builder,
            args[0],
            signature.args[0],
            signature.return_type,
        )

    return sig, codegen


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


if sys.version_info >= (3, 11):

    @overload(math.exp2, target="cuda")
    def exp2_ol(a):
        return _make_unary(a, hexp2)


## Public aliases using Numba/Numpy-style type names
# Floating-point
float32_to_bfloat16 = float2bfloat16
float64_to_bfloat16 = double2bfloat16
bfloat16_to_float32 = bfloat162float
float32_to_bfloat16_rn = float2bfloat16_rn
float32_to_bfloat16_rz = float2bfloat16_rz
float32_to_bfloat16_rd = float2bfloat16_rd
float32_to_bfloat16_ru = float2bfloat16_ru

# Char (8-bit)
bfloat16_to_int8_rz = bfloat162char_rz
bfloat16_to_uint8_rz = bfloat162uchar_rz

# Int16 / UInt16
int16_to_bfloat16_rn = short2bfloat16_rn
int16_to_bfloat16_rz = short2bfloat16_rz
int16_to_bfloat16_rd = short2bfloat16_rd
int16_to_bfloat16_ru = short2bfloat16_ru
bfloat16_to_int16_rn = bfloat162short_rn
bfloat16_to_int16_rz = bfloat162short_rz
bfloat16_to_int16_rd = bfloat162short_rd
bfloat16_to_int16_ru = bfloat162short_ru

uint16_to_bfloat16_rn = ushort2bfloat16_rn
uint16_to_bfloat16_rz = ushort2bfloat16_rz
uint16_to_bfloat16_rd = ushort2bfloat16_rd
uint16_to_bfloat16_ru = ushort2bfloat16_ru
bfloat16_to_uint16_rn = bfloat162ushort_rn
bfloat16_to_uint16_rz = bfloat162ushort_rz
bfloat16_to_uint16_rd = bfloat162ushort_rd
bfloat16_to_uint16_ru = bfloat162ushort_ru

# Int32 / UInt32
int32_to_bfloat16_rn = int2bfloat16_rn
int32_to_bfloat16_rz = int2bfloat16_rz
int32_to_bfloat16_rd = int2bfloat16_rd
int32_to_bfloat16_ru = int2bfloat16_ru
bfloat16_to_int32_rn = bfloat162int_rn
bfloat16_to_int32_rz = bfloat162int_rz
bfloat16_to_int32_rd = bfloat162int_rd
bfloat16_to_int32_ru = bfloat162int_ru

uint32_to_bfloat16_rn = uint2bfloat16_rn
uint32_to_bfloat16_rz = uint2bfloat16_rz
uint32_to_bfloat16_rd = uint2bfloat16_rd
uint32_to_bfloat16_ru = uint2bfloat16_ru
bfloat16_to_uint32_rn = bfloat162uint_rn
bfloat16_to_uint32_rz = bfloat162uint_rz
bfloat16_to_uint32_rd = bfloat162uint_rd
bfloat16_to_uint32_ru = bfloat162uint_ru

# Int64 / UInt64
int64_to_bfloat16_rn = ll2bfloat16_rn
int64_to_bfloat16_rz = ll2bfloat16_rz
int64_to_bfloat16_rd = ll2bfloat16_rd
int64_to_bfloat16_ru = ll2bfloat16_ru
bfloat16_to_int64_rn = bfloat162ll_rn
bfloat16_to_int64_rz = bfloat162ll_rz
bfloat16_to_int64_rd = bfloat162ll_rd
bfloat16_to_int64_ru = bfloat162ll_ru

uint64_to_bfloat16_rn = ull2bfloat16_rn
uint64_to_bfloat16_rz = ull2bfloat16_rz
uint64_to_bfloat16_rd = ull2bfloat16_rd
uint64_to_bfloat16_ru = ull2bfloat16_ru
bfloat16_to_uint64_rn = bfloat162ull_rn
bfloat16_to_uint64_rz = bfloat162ull_rz
bfloat16_to_uint64_rd = bfloat162ull_rd
bfloat16_to_uint64_ru = bfloat162ull_ru

# Bit reinterpret casts
bfloat16_as_int16 = bfloat16_as_short
bfloat16_as_uint16 = bfloat16_as_ushort
int16_as_bfloat16 = short_as_bfloat16
uint16_as_bfloat16 = ushort_as_bfloat16

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
    "float32_to_bfloat16",
    "float64_to_bfloat16",
    "bfloat16_to_float32",
    "float32_to_bfloat16_rn",
    "float32_to_bfloat16_rz",
    "float32_to_bfloat16_rd",
    "float32_to_bfloat16_ru",
    "bfloat16_to_int8_rz",
    "bfloat16_to_uint8_rz",
    "int16_to_bfloat16_rn",
    "int16_to_bfloat16_rz",
    "int16_to_bfloat16_rd",
    "int16_to_bfloat16_ru",
    "bfloat16_to_int16_rn",
    "bfloat16_to_int16_rz",
    "bfloat16_to_int16_rd",
    "bfloat16_to_int16_ru",
    "uint16_to_bfloat16_rn",
    "uint16_to_bfloat16_rz",
    "uint16_to_bfloat16_rd",
    "uint16_to_bfloat16_ru",
    "bfloat16_to_uint16_rn",
    "bfloat16_to_uint16_rz",
    "bfloat16_to_uint16_rd",
    "bfloat16_to_uint16_ru",
    "int32_to_bfloat16_rn",
    "int32_to_bfloat16_rz",
    "int32_to_bfloat16_rd",
    "int32_to_bfloat16_ru",
    "bfloat16_to_int32_rn",
    "bfloat16_to_int32_rz",
    "bfloat16_to_int32_rd",
    "bfloat16_to_int32_ru",
    "uint32_to_bfloat16_rn",
    "uint32_to_bfloat16_rz",
    "uint32_to_bfloat16_rd",
    "uint32_to_bfloat16_ru",
    "bfloat16_to_uint32_rn",
    "bfloat16_to_uint32_rz",
    "bfloat16_to_uint32_rd",
    "bfloat16_to_uint32_ru",
    "int64_to_bfloat16_rn",
    "int64_to_bfloat16_rz",
    "int64_to_bfloat16_rd",
    "int64_to_bfloat16_ru",
    "bfloat16_to_int64_rn",
    "bfloat16_to_int64_rz",
    "bfloat16_to_int64_rd",
    "bfloat16_to_int64_ru",
    "uint64_to_bfloat16_rn",
    "uint64_to_bfloat16_rz",
    "uint64_to_bfloat16_rd",
    "uint64_to_bfloat16_ru",
    "bfloat16_to_uint64_rn",
    "bfloat16_to_uint64_rz",
    "bfloat16_to_uint64_rd",
    "bfloat16_to_uint64_ru",
    "bfloat16_as_int16",
    "bfloat16_as_uint16",
    "int16_as_bfloat16",
    "uint16_as_bfloat16",
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
