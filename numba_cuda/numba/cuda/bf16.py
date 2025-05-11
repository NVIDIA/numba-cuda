from numba.cuda._internal.cuda_bf16 import (
    _type_class___nv_bfloat16,
    nv_bfloat16 as bfloat16,
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
    if isinstance(a, _type_class___nv_bfloat16):
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
    "bfloat16",
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
