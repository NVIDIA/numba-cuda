# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import threading
import time

from numba import cuda
import numba.cuda.dispatcher as dispatcher_module
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
        cfg.dispatcher.mark_launch_config_sensitive()
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

    def test_concurrent_launch_config_specialization_compiles_once(self):
        @cuda.jit
        def launch_config_sensitive_kernel(x):
            x[0] = 1

        arr = np.zeros(1, dtype=np.int32)
        launch_config_sensitive_kernel[1, 32](arr)
        self.assertTrue(launch_config_sensitive_kernel._launch_config_sensitive)
        self.assertEqual(len(LAUNCH_CONFIG_LOG), 1)

        original_init = dispatcher_module.CUDADispatcher.__init__
        constructed_dispatchers = []
        constructed_lock = threading.Lock()

        def counting_init(self, *args, **kwargs):
            with constructed_lock:
                constructed_dispatchers.append(threading.get_ident())
            # Widen the race window so concurrent launches all see the missing
            # launch-config specialization before the first one can finish
            # construction.
            time.sleep(0.05)
            return original_init(self, *args, **kwargs)

        barrier = threading.Barrier(16)
        errors = []
        errors_lock = threading.Lock()

        def launch():
            try:
                local = np.zeros(1, dtype=np.int32)
                barrier.wait(timeout=10)
                launch_config_sensitive_kernel[1, 64](local)
                self.assertEqual(local[0], 1)
            except BaseException as e:
                with errors_lock:
                    errors.append(e)

        threads = [threading.Thread(target=launch) for _ in range(16)]
        dispatcher_module.CUDADispatcher.__init__ = counting_init
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)
        finally:
            dispatcher_module.CUDADispatcher.__init__ = original_init

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            len(launch_config_sensitive_kernel._launch_config_specializations),
            1,
        )
        self.assertEqual(len(constructed_dispatchers), 1)
        self.assertEqual(len(LAUNCH_CONFIG_LOG), 2)
        self.assertEqual(LAUNCH_CONFIG_LOG[1]["blockdim"], (64, 1, 1))

    def test_concurrent_first_launches_record_sensitivity_before_overload(
        self,
    ):
        @cuda.jit
        def launch_config_sensitive_kernel(x):
            x[0] = 1

        original_update = (
            dispatcher_module.CUDADispatcher._update_launch_config_sensitivity
        )
        first_update_entered = threading.Event()
        release_update = threading.Event()
        blocked_once = threading.Event()

        def blocking_update(dispatcher, kernel, launch_config):
            if (
                dispatcher is launch_config_sensitive_kernel
                and getattr(kernel, "launch_config_sensitive", False)
                and dispatcher._launch_config_default_key is None
                and not blocked_once.is_set()
            ):
                blocked_once.set()
                first_update_entered.set()
                if not release_update.wait(timeout=10):
                    raise AssertionError(
                        "timed out waiting to release LCS update"
                    )
            return original_update(dispatcher, kernel, launch_config)

        barrier = threading.Barrier(2)
        errors = []
        errors_lock = threading.Lock()

        def launch(blockdim):
            try:
                local = np.zeros(1, dtype=np.int32)
                barrier.wait(timeout=10)
                launch_config_sensitive_kernel[1, blockdim](local)
                self.assertEqual(local[0], 1)
            except BaseException as e:
                with errors_lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=launch, args=(blockdim,))
            for blockdim in (32, 64)
        ]

        dispatcher_module.CUDADispatcher._update_launch_config_sensitivity = (
            blocking_update
        )
        try:
            for thread in threads:
                thread.start()
            update_entered = first_update_entered.wait(timeout=20)
            # Give the second thread time to reach the same dispatcher while
            # the first launch is in the LCS publication window.
            time.sleep(0.05)
            release_update.set()
            for thread in threads:
                thread.join(timeout=20)
        finally:
            release_update.set()
            dispatcher_class = dispatcher_module.CUDADispatcher
            dispatcher_class._update_launch_config_sensitivity = original_update

        self.assertTrue(update_entered)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(LAUNCH_CONFIG_LOG), 2)
        self.assertEqual(
            {entry["blockdim"] for entry in LAUNCH_CONFIG_LOG},
            {(32, 1, 1), (64, 1, 1)},
        )

    def test_concurrent_distinct_launch_config_specializations(self):
        @cuda.jit
        def launch_config_sensitive_kernel(x, mult):
            i = cuda.grid(1)
            if i < x.size:
                x[i] *= mult[0]

        configs = ((1, 32), (1, 64), (2, 32), (2, 64))
        barrier = threading.Barrier(len(configs))
        errors = []
        errors_lock = threading.Lock()

        def launch(config):
            try:
                blocks, threads = config
                n = blocks * threads
                arr = cuda.to_device(np.ones(n, dtype=np.int32))
                mult = cuda.to_device(np.array([2], dtype=np.int32))
                barrier.wait(timeout=10)
                launch_config_sensitive_kernel[blocks, threads](arr, mult)
                np.testing.assert_array_equal(
                    arr.copy_to_host(), np.full(n, 2, dtype=np.int32)
                )
            except BaseException as e:
                with errors_lock:
                    errors.append(e)

        threads = [threading.Thread(target=launch, args=(c,)) for c in configs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertTrue(launch_config_sensitive_kernel._launch_config_sensitive)
        self.assertIsNotNone(
            launch_config_sensitive_kernel._launch_config_default_key
        )
        self.assertLessEqual(
            len(launch_config_sensitive_kernel._launch_config_specializations),
            len(configs),
        )
        log_configs = {
            (entry["griddim"], entry["blockdim"]) for entry in LAUNCH_CONFIG_LOG
        }
        self.assertEqual(
            log_configs,
            {((b, 1, 1), (t, 1, 1)) for b, t in configs},
        )


if __name__ == "__main__":
    unittest.main()
