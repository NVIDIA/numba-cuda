# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import importlib
from numba.cuda.core import config
from .utils import _readenv
import warnings
import sys


# Enable pynvjitlink based on the following precedence:
# 1. Config setting "CUDA_ENABLE_PYNVJITLINK" (highest priority)
# 2. Environment variable "NUMBA_CUDA_ENABLE_PYNVJITLINK"
# 3. Auto-detection of pynvjitlink module (lowest priority)

pynvjitlink_auto_enabled = False

if getattr(config, "CUDA_ENABLE_PYNVJITLINK", None) is None:
    if (
        _pynvjitlink_enabled_in_env := _readenv(
            "NUMBA_CUDA_ENABLE_PYNVJITLINK", bool, None
        )
    ) is not None:
        config.CUDA_ENABLE_PYNVJITLINK = _pynvjitlink_enabled_in_env
    else:
        pynvjitlink_auto_enabled = (
            importlib.util.find_spec("pynvjitlink") is not None
        )
        config.CUDA_ENABLE_PYNVJITLINK = pynvjitlink_auto_enabled

# Upstream numba sets CUDA_USE_NVIDIA_BINDING to 0 by default, so it always
# exists. Override, but not if explicitly set to 0 in the envioronment.
_nvidia_binding_enabled_in_env = _readenv(
    "NUMBA_CUDA_USE_NVIDIA_BINDING", bool, None
)
if _nvidia_binding_enabled_in_env is False:
    USE_NV_BINDING = False
else:
    USE_NV_BINDING = True
    config.CUDA_USE_NVIDIA_BINDING = USE_NV_BINDING
if config.CUDA_USE_NVIDIA_BINDING:
    if not (
        importlib.util.find_spec("cuda")
        and importlib.util.find_spec("cuda.bindings")
    ):
        raise ImportError(
            "CUDA bindings not found. Please pip install the "
            "cuda-bindings package. Alternatively, install "
            "numba-cuda[cuXY], where XY is the required CUDA "
            "version, to install the binding automatically. "
            "If no CUDA bindings are desired, set the env var "
            "NUMBA_CUDA_USE_NVIDIA_BINDING=0 to enable ctypes "
            "bindings."
        )

if config.CUDA_ENABLE_PYNVJITLINK:
    if USE_NV_BINDING and not pynvjitlink_auto_enabled:
        warnings.warn(
            "Explicitly enabling pynvjitlink is no longer necessary. "
            "NVIDIA bindings are enabled. cuda.core will be used "
            "in place of pynvjitlink."
        )
    elif pynvjitlink_auto_enabled:
        # Ignore the fact that pynvjitlink is enabled, because that was an
        # automatic decision based on discovering pynvjitlink was present; the
        # user didn't ask for it
        pass
    else:
        raise RuntimeError("nvJitLink requires the NVIDIA CUDA bindings. ")

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
