# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import concurrent.futures
import multiprocessing
import os
from numba.cuda.testing import unittest


def set_visible_devices_and_check():
    from numba import cuda
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return len(cuda.gpus.lst)


class TestVisibleDevices(unittest.TestCase):
    def test_visible_devices_set_after_import(self):
        # See Issue #6149. This test checks that we can set
        # CUDA_VISIBLE_DEVICES after importing Numba and have the value
        # reflected in the available list of GPUs. Prior to the fix for this
        # issue, Numba made a call to runtime.get_version() on import that
        # initialized the driver and froze the list of available devices before
        # CUDA_VISIBLE_DEVICES could be set by the user.

        # Avoid importing cuda at the top level so that
        # set_visible_devices_and_check gets to import it first in its process
        from numba import cuda

        if len(cuda.gpus.lst) in (0, 1):
            self.skipTest("This test requires multiple GPUs")

        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            msg = "Cannot test when CUDA_VISIBLE_DEVICES already set"
            self.skipTest(msg)

        with concurrent.futures.ProcessPoolExecutor(
            mp_context=multiprocessing.get_context("spawn")
        ) as exe:
            future = exe.submit(set_visible_devices_and_check)

        visible_gpu_count = future.result()
        assert visible_gpu_count == 1


if __name__ == "__main__":
    unittest.main()
