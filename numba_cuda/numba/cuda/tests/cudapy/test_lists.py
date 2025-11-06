# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Test cases for a list object within CUDA kernels
"""

from numba import cuda
from numba.cuda import config
import numpy as np
from numba.cuda.testing import (
    CUDATestCase,
)


class ListTest(CUDATestCase):
    def setUp(self):
        self.old_nrt_setting = config.CUDA_ENABLE_NRT
        config.CUDA_ENABLE_NRT = True
        super().setUp()

    def tearDown(self):
        config.CUDA_ENABLE_NRT = self.old_nrt_setting
        super().tearDown()

    def test_list_roundtrip(self):
        lst = [1, 2, 3]

        @cuda.jit
        def kernel(out):
            for i in range(len(lst)):
                out[i] = lst[i]

        out = cuda.to_device(np.zeros(len(lst)))

        kernel[1, 1](out)
        for g, e in zip(out.copy_to_host(), lst):
            self.assertEqual(e, g)
