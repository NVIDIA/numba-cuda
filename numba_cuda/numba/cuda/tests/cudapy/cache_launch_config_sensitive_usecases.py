# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda import launchconfig
from numba.cuda.core.rewrites import register_rewrite, Rewrite, rewrite_registry
import sys

_REWRITE_FLAG = "_launch_config_cache_rewrite_registered"


if not getattr(rewrite_registry, _REWRITE_FLAG, False):

    @register_rewrite("after-inference")
    class LaunchConfigSensitiveCacheRewrite(Rewrite):
        _TARGET_NAME = "lcs_cache_kernel"

        def __init__(self, state):
            super().__init__(state)
            self._state = state
            self._block = None
            self._logged = False

        def match(self, func_ir, block, typemap, calltypes):
            if func_ir.func_id.func_name != self._TARGET_NAME:
                return False
            if self._logged:
                return False
            self._block = block
            return True

        def apply(self):
            # Ensure launch config is available and mark compilation as
            # launch-config sensitive so the disk cache keys include it.
            launchconfig.ensure_current_launch_config()
            self._state.metadata["launch_config_sensitive"] = True
            self._logged = True
            return self._block

    setattr(rewrite_registry, _REWRITE_FLAG, True)


@cuda.jit(cache=True)
def lcs_cache_kernel(x):
    x[0] = 1


def launch(blockdim):
    arr = np.zeros(1, dtype=np.int32)
    lcs_cache_kernel[1, blockdim](arr)
    return arr


def self_test():
    mod = sys.modules[__name__]
    out = mod.launch(32)
    assert out[0] == 1
    out = mod.launch(64)
    assert out[0] == 1
