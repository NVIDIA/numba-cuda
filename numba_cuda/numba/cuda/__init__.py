# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import importlib
from numba.cuda.core import config
from .utils import _readenv
import warnings
import sys

# Import version from the parent package
from numba_cuda._version import __version__

# Generate version_info early to avoid circular import issues
from numba_cuda._version import generate_version_info

version_info = generate_version_info(__version__)
del generate_version_info

# Re-export types itself
import numba.cuda.types as types

# Re-export all type names
from numba.cuda.types import *

# Check if upstream numba is available - needs to be defined early to avoid circular imports
try:
    import numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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
from numba.cuda.misc import quicksort, mergesort
from numba.cuda.misc.special import literal_unroll
from numba.cuda.np.numpy_support import *
