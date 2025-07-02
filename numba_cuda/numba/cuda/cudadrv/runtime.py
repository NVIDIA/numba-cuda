"""
Former CUDA Runtime wrapper.

The toolkit version can now be obtained from NVRTC, so we don't use a binding
to the runtime anymore. This file is provided to maintain the existing API.
"""

from numba.cuda.cudadrv.nvrtc import NVRTC


class Runtime:
    def get_version(self):
        return NVRTC().get_version()


runtime = Runtime()


def get_version():
    """
    Return the runtime version as a tuple of (major, minor)
    """
    return runtime.get_version()
