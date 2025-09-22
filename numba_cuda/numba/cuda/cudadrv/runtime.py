# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Former CUDA Runtime wrapper.

The toolkit version can now be obtained from NVRTC, so we don't use a binding
to the runtime anymore. This file is provided to maintain the existing API.
"""

from numba.cuda.cudadrv.nvrtc import _get_nvrtc_version


class Runtime:
    def get_version(self):
        return _get_nvrtc_version()


runtime = Runtime()


def get_version():
    """
    Return the runtime version as a tuple of (major, minor)
    """
    return runtime.get_version()
