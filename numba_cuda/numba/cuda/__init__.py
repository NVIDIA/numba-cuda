from numba.core import config

from numba import runtests
from numba.cuda.compiler import (
    compile,
    compile_for_current_device,
    compile_ptx,
    compile_ptx_for_current_device,
)
from .utils import _readenv


if config.ENABLE_CUDASIM:
    from .simulator_init import *
else:
    from .device_init import *
    from .device_init import _auto_device


# This is the out-of-tree NVIDIA-maintained target. This is reported in Numba
# sysinfo (`numba -s`):
implementation = "NVIDIA"


def test(*args, **kwargs):
    if not is_available():
        raise cuda_error()

    return runtests.main("numba.cuda.tests", *args, **kwargs)
