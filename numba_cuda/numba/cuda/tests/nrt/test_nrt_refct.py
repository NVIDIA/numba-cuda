# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import unittest
from numba.cuda.tests.support import override_config
from numba.cuda.memory_management import rtsys
from numba.cuda.tests.support import EnableNRTStatsMixin
from numba.cuda.testing import CUDATestCase, skip_on_cudasim

from numba import cuda


@skip_on_cudasim("No refcounting in the simulator")
class TestNrtRefCt(EnableNRTStatsMixin, CUDATestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def run(self, result=None):
        with (
            override_config("CUDA_ENABLE_NRT", True),
            override_config("CUDA_NRT_STATS", True),
        ):
            super().run(result)

    def test_no_return(self):
        """
        Test issue #1291
        """

        n = 10

        @cuda.jit
        def kernel():
            for i in range(n):
                temp = np.empty(2)  # noqa: F841
            return None

        init_stats = rtsys.get_allocation_stats()
        kernel[1, 1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)

    def test_escaping_var_init_in_loop(self):
        """
        Test issue #1297
        """

        @cuda.jit
        def g(n):
            x = np.empty((n, 2))

            for i in range(n):
                y = x[i]

            for i in range(n):
                y = x[i]  # noqa: F841

            return None

        init_stats = rtsys.get_allocation_stats()
        g[1, 1](10)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, 1)
        self.assertEqual(cur_stats.free - init_stats.free, 1)

    def test_invalid_computation_of_lifetime(self):
        """
        Test issue #1573
        """

        @cuda.jit
        def if_with_allocation_and_initialization(arr1, test1):
            tmp_arr = np.empty_like(arr1)

            for i in range(tmp_arr.shape[0]):
                pass

            if test1:
                np.empty_like(arr1)

        arr = np.random.random((5, 5))  # the values are not consumed

        init_stats = rtsys.get_allocation_stats()
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            if_with_allocation_and_initialization[1, 1](arr, False)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(
            cur_stats.alloc - init_stats.alloc, cur_stats.free - init_stats.free
        )

    def test_del_at_beginning_of_loop(self):
        """
        Test issue #1734
        """

        @cuda.jit
        def f(arr):
            res = 0

            for i in (0, 1):
                # `del t` is issued here before defining t.  It must be
                # correctly handled by the lowering phase.
                t = arr[i]
                if t[i] > 1:
                    res += t[i]

        arr = np.ones((2, 2))

        init_stats = rtsys.get_allocation_stats()
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            f[1, 1](arr)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(
            cur_stats.alloc - init_stats.alloc, cur_stats.free - init_stats.free
        )


if __name__ == "__main__":
    unittest.main()
