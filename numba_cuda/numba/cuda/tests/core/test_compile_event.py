# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from numba import cuda
from numba.cuda.core import event as ev
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


class _Recorder(ev.Listener):
    def __init__(self):
        self.starts = []
        self.ends = []

    def on_start(self, event):
        self.starts.append(event)

    def on_end(self, event):
        self.ends.append(event)


class TestCompileEventKind(unittest.TestCase):
    # Whatever kind the dispatcher passes to ``trigger_event`` has to survive
    # ``_guard_kind``, which runs on context entry. A rejected kind raises
    # before the kernel is built, so a wrong prefix turns every cache miss
    # into a ValueError.
    def test_reserved_numba_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            ev.start_event("numba:compile")

    def test_builtin_compile_kind_is_accepted(self):
        ev.start_event("numba-cuda:compile")
        ev.end_event("numba-cuda:compile")


@skip_on_cudasim("the dispatcher under test is not used by the simulator")
class TestCompileEventEmitted(CUDATestCase):
    def test_kernel_compile_emits_start_and_end(self):
        @cuda.jit
        def kernel(x):
            x[0] = 1

        rec = _Recorder()
        ev.register("numba-cuda:compile", rec)
        try:
            kernel[1, 1](cuda.device_array(1, dtype="int64"))
        finally:
            ev.unregister("numba-cuda:compile", rec)

        self.assertTrue(rec.starts)
        self.assertTrue(rec.ends)
        self.assertIsInstance(rec.starts[0].data, dict)
        self.assertIn("dispatcher", rec.starts[0].data)


if __name__ == "__main__":
    unittest.main()
