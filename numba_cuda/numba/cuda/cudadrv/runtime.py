# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Former CUDA Runtime wrapper.

The toolkit version can now be obtained from NVRTC, so we don't use a binding
to the runtime anymore. This file is provided to maintain the existing API.
"""

from numba import config
from numba.cuda.cudadrv.nvrtc import NVRTC


class Runtime:
    def get_version(self):
        if config.CUDA_USE_NVIDIA_BINDING:
            from cuda.bindings import nvrtc

            retcode, *version = nvrtc.nvrtcVersion()
            if retcode != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(
                    f"{retcode.name} when calling nvrtcGetVersion()"
                )
            return tuple(version)
        else:
            return NVRTC().get_version()


runtime = Runtime()


def get_version():
    """
    Return the runtime version as a tuple of (major, minor)
    """
    return runtime.get_version()
