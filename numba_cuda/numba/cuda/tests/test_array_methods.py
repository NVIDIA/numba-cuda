# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import cuda
from numba.cuda import config


class TestArrayMethods(MemoryLeakMixin, TestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    def setUp(self):
        super(TestArrayMethods, self).setUp()
        np.random.seed(42)
        self.old_nrt_setting = config.CUDA_ENABLE_NRT
        config.CUDA_ENABLE_NRT = True

    def tearDown(self):
        config.CUDA_ENABLE_NRT = self.old_nrt_setting
        super(TestArrayMethods, self).tearDown()

    def test_array_copy(self):
        ary = np.array([1.0, 2.0, 3.0])
        out = cuda.to_device(np.zeros(3))

        @cuda.jit
        def kernel(out):
            gid = cuda.grid(1)
            if gid < 1:
                cpy = ary.copy()
                for i in range(len(out)):
                    out[i] = cpy[i]

        kernel[1, 1](out)

        result = out.copy_to_host()
        np.testing.assert_array_equal(result, ary)
