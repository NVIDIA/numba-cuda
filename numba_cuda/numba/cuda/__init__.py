# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import importlib
from numba.cuda.core import config
from .utils import _readenv
import warnings
import sys

# Re-export types itself
import numba.cuda.types as types

# Re-export all type names
from numba.cuda.types import *


# Require NVIDIA CUDA bindings at import time
if not (
    importlib.util.find_spec("cuda")
    and importlib.util.find_spec("cuda.bindings")
):
    raise ImportError(
        "NVIDIA CUDA Python bindings not found. Install the 'cuda' package "
        "(e.g. pip install nvidia-cuda-python or numba-cuda[cuXY])."
    )

if config.ENABLE_CUDASIM:
    from .simulator_init import *
else:
    from .device_init import *
    from .device_init import _auto_device

from numba.cuda.compiler import (
    compile,
    compile_for_current_device,
    compile_ptx,
    compile_ptx_for_current_device,
    compile_all,
)

# This is the out-of-tree NVIDIA-maintained target. This is reported in Numba
# sysinfo (`numba -s`):
implementation = "NVIDIA"


# The default compute capability as set by the upstream Numba implementation.
config_default_cc = config.CUDA_DEFAULT_PTX_CC

# The default compute capability for Numba-CUDA. This will usually override the
# upstream Numba built-in default of 5.0, unless the user has set it even
# higher, in which case we should use the user-specified value. This default is
# aligned with recent toolkit versions.
numba_cuda_default_ptx_cc = (7, 5)

if numba_cuda_default_ptx_cc > config_default_cc:
    config.CUDA_DEFAULT_PTX_CC = numba_cuda_default_ptx_cc


# Warn if on Linux and RTLD_GLOBAL is enabled
if sys.platform.startswith("linux") and (sys.getdlopenflags() & 0x100) != 0:
    warnings.warn(
        "RTLD_GLOBAL is enabled, which might result in symbol resolution "
        "conflicts when importing both numba and numba.cuda. Consider using "
        "sys.setdlopenflags() to disable RTLD_GLOBAL "
        "if you encounter symbol conflicts."
    )

from numba.cuda.np.ufunc import vectorize, guvectorize


_lazy_exports = {
    "literal_unroll": ("numba.cuda.misc.special", "literal_unroll"),
    "literal": ("numba.cuda.misc", "literal"),
}

__all__ = list(globals().get("__all__", [])) + list(_lazy_exports.keys())


def __getattr__(name):
    """
    Lazily import a few attrs that might import lowering
    """
    if name in _lazy_exports:
        mod_name, attr_name = _lazy_exports[name]
        try:
            module = importlib.import_module(mod_name)
            value = getattr(module, attr_name)
        except Exception as exc:
            raise ImportError(
                f"could not import name {attr_name!r} from {mod_name!r}"
            ) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_lazy_exports.keys())))
