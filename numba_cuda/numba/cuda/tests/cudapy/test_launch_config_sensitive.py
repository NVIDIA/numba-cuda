# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda import launchconfig
from numba.cuda.core.rewrites import register_rewrite, Rewrite
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase


LAUNCH_CONFIG_LOG = []


def _clear_launch_config_log():
    LAUNCH_CONFIG_LOG.clear()


@register_rewrite("after-inference")
class LaunchConfigSensitiveRewrite(Rewrite):
    """Rewrite that marks kernels as launch-config sensitive and logs config.

    This mimics cuda.coop's need to access launch config during rewrite, and
    provides a global log for tests to assert on.
    """

    _TARGET_NAME = "launch_config_sensitive_kernel"

    def __init__(self, state):
        super().__init__(state)
        self._state = state
        self._logged = False
        self._block = None

    def match(self, func_ir, block, typemap, calltypes):
        if func_ir.func_id.func_name != self._TARGET_NAME:
            return False
        if self._logged:
            return False
        self._block = block
        return True

    def apply(self):
        cfg = launchconfig.ensure_current_launch_config()
        LAUNCH_CONFIG_LOG.append(
            {
                "griddim": cfg.griddim,
                "blockdim": cfg.blockdim,
                "sharedmem": cfg.sharedmem,
            }
        )
        # Mark compilation as launch-config sensitive so the dispatcher can
        # avoid reusing the compiled kernel across different launch configs.
        cfg.mark_kernel_as_launch_config_sensitive()
        self._logged = True
        return self._block


@skip_on_cudasim("Dispatcher C-extension not used in the simulator")
class TestLaunchConfigSensitive(CUDATestCase):
    def setUp(self):
        super().setUp()
        _clear_launch_config_log()

    def test_launch_config_sensitive_requires_recompile(self):
        @cuda.jit
        def launch_config_sensitive_kernel(x):
            x[0] = 1

        arr = np.zeros(1, dtype=np.int32)

        launch_config_sensitive_kernel[1, 32](arr)
        self.assertEqual(len(LAUNCH_CONFIG_LOG), 1)
        self.assertEqual(LAUNCH_CONFIG_LOG[0]["blockdim"], (32, 1, 1))
        self.assertEqual(LAUNCH_CONFIG_LOG[0]["griddim"], (1, 1, 1))

        launch_config_sensitive_kernel[1, 64](arr)
        # Expect a new compilation for the new launch config, which will log
        # a second entry with the updated block dimension.
        self.assertEqual(len(LAUNCH_CONFIG_LOG), 2)
        self.assertEqual(LAUNCH_CONFIG_LOG[1]["blockdim"], (64, 1, 1))
        self.assertEqual(LAUNCH_CONFIG_LOG[1]["griddim"], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
