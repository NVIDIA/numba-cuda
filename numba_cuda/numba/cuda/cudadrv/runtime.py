# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Former CUDA Runtime wrapper.

The toolkit version can now be obtained from NVRTC, so we don't use a binding
to the runtime anymore. This file is provided to maintain the existing API.
"""

from numba.cuda.cudadrv.nvrtc import _get_nvrtc_version

SUPPORTED_TOOLKIT_MAJOR_VERSIONS = (12, 13)


class Runtime:
    def get_version(self):
        return _get_nvrtc_version()

    def is_supported_version(self):
        """
        Returns True if the CUDA Runtime is a supported version.
        """
        return self.get_version()[0] in SUPPORTED_TOOLKIT_MAJOR_VERSIONS


runtime = Runtime()


def get_version():
    """
    Return the runtime version as a tuple of (major, minor)
    """
    return runtime.get_version()
