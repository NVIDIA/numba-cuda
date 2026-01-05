# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest
from numba.cuda.testing import CUDATestCase
from numba import cuda
from numba.cuda.testing import skip_on_cudasim
import cupy as cp


@skip_on_cudasim("CUDA Profiler unsupported in the simulator")
class TestProfiler(CUDATestCase):
    def test_profiling(self):
        with cuda.profiling():
            a = cp.zeros(10)
            del a

        with cuda.profiling():
            a = cp.zeros(100)
            del a


if __name__ == "__main__":
    unittest.main()
