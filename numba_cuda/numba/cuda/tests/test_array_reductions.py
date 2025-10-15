# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import cuda


class TestArrayReductions(MemoryLeakMixin, TestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    def setUp(self):
        super(TestArrayReductions, self).setUp()
        np.random.seed(42)

    def test_all_basic(self):
        def check(arr):
            @cuda.jit
            def kernel(out):
                gid = cuda.grid(1)
                if gid < 1:
                    out[0] = np.all(arr)

            out = cuda.to_device(np.zeros(1, dtype=np.bool_))
            kernel[1, 1](out)
            self.assertPreciseEqual(np.all(arr), out.copy_to_host()[0])

        arr = np.float64([1.0, 0.0, float("inf"), float("nan")])
        check(arr)
        arr = np.float64([1.0, -0.0, float("inf"), float("nan")])
        check(arr)
        arr = np.float64([1.0, 1.5, float("inf"), float("nan")])
        check(arr)
        arr = np.float64([[1.0, 1.5], [float("inf"), float("nan")]])
        check(arr)
        arr = np.float64([[1.0, 1.5], [1.5, 1.0]])
        check(arr)
