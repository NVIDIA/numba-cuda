import importlib
from numba import runtests
from numba.core import config
from .utils import _readenv

# Enable pynvjitlink if the environment variables NUMBA_CUDA_ENABLE_PYNVJITLINK
# or CUDA_ENABLE_PYNVJITLINK are set, or if the pynvjitlink module is found. If
# explicitly disabled, do not use pynvjitlink, even if present in the env.
_pynvjitlink_enabled_in_env = _readenv(
    "NUMBA_CUDA_ENABLE_PYNVJITLINK", bool, None
)
_pynvjitlink_enabled_in_cfg = getattr(config, "CUDA_ENABLE_PYNVJITLINK", None)

if _pynvjitlink_enabled_in_env is not None:
    ENABLE_PYNVJITLINK = _pynvjitlink_enabled_in_env
elif _pynvjitlink_enabled_in_cfg is not None:
    ENABLE_PYNVJITLINK = _pynvjitlink_enabled_in_cfg
else:
    ENABLE_PYNVJITLINK = importlib.util.find_spec("pynvjitlink") is not None

if not hasattr(config, "CUDA_ENABLE_PYNVJITLINK"):
    config.CUDA_ENABLE_PYNVJITLINK = ENABLE_PYNVJITLINK

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
)

# This is the out-of-tree NVIDIA-maintained target. This is reported in Numba
# sysinfo (`numba -s`):
implementation = "NVIDIA"


def test(*args, **kwargs):
    if not is_available():
        raise cuda_error()

    return runtests.main("numba.cuda.tests", *args, **kwargs)
