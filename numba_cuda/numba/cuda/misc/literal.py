# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.extending import overload
from numba.cuda import types
from numba.cuda.misc.special import literally, literal_unroll
from numba.cuda.core.errors import TypingError


@overload(literally, target="cuda")
def _ov_literally(obj):
    if isinstance(obj, (types.Literal, types.InitialValue)):
        return lambda obj: obj
    else:
        m = "Invalid use of non-Literal type in literally({})".format(obj)
        raise TypingError(m)


@overload(literal_unroll, target="cuda")
def literal_unroll_impl(container):
    if isinstance(container, types.Poison):
        m = f"Invalid use of non-Literal type in literal_unroll({container})"
        raise TypingError(m)

    def impl(container):
        return container

    return impl
