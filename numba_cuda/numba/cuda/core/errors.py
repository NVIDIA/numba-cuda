# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
from numba.cuda.utils import redirect_numba_module

_mod = redirect_numba_module(
    locals(), "numba.core.errors", "numba.cuda.core.cuda_errors"
)

# ---------------------------------------------------------------------------
# Restore the pre-redirect exception class hierarchy  (issue #755)
# ---------------------------------------------------------------------------
#
# Before commit 491f552 ("Always use core errors if numba is present"),
# this module defined its own exception classes as *subclasses* of the
# upstream numba.core.errors classes.  This meant
# ``isinstance(upstream_exc, cuda_NumbaError)`` was False -- the CUDA
# compiler relied on this narrowness to short-circuit template search
# and pipeline retries.  The redirect replaced the subclass hierarchy
# with identity (every name became an alias for the upstream class),
# broadening every isinstance check and causing orders-of-magnitude
# slower compile times for types with many overload candidates (e.g.
# strings).
#
# The fix uses dynamic diamond inheritance: for every upstream
# NumbaError descendant, a local subclass is created that inherits
# from both the local parent and the upstream class.  This means:
#
#   isinstance(cuda_TypingError, core_TypingError) -> True
#       (user except clauses still catch CUDA exceptions)
#   isinstance(core_TypingError, cuda_NumbaError)  -> False
#       (narrow isinstance gate restored for compiler internals)

try:
    import numba.core.errors as _ce

    class NumbaError(_ce.NumbaError):
        pass

    # Build local subclasses for every upstream NumbaError descendant.
    # Process parents before children: repeat until every class whose
    # parent has already been remapped is handled.
    _remap = {_ce.NumbaError: NumbaError}
    _remaining = [
        (name, obj)
        for name, obj in vars(_ce).items()
        if isinstance(obj, type)
        and issubclass(obj, _ce.NumbaError)
        and obj is not _ce.NumbaError
    ]
    while _remaining:
        _next_round = []
        for _name, _obj in _remaining:
            # Find the first base class already in the remap.  This
            # handles any base ordering and mixins that aren't part of
            # the NumbaError hierarchy.
            _parent = None
            for _base in _obj.__bases__:
                if _base in _remap:
                    _parent = _remap[_base]
                    break
            if _parent is not None:
                _local = type(_name, (_parent, _obj), {})
                _remap[_obj] = _local
                setattr(_mod, _name, _local)
            else:
                _next_round.append((_name, _obj))
        if len(_next_round) == len(_remaining):
            break  # no progress — avoid infinite loop
        _remaining = _next_round

    setattr(_mod, "NumbaError", NumbaError)

except ImportError:
    pass

sys.modules[__name__] = _mod
