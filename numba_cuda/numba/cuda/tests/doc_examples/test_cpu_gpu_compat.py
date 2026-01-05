# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from numba.cuda.testing import (
    CUDATestCase,
    skip_on_cudasim,
    skip_on_standalone_numba_cuda,
)
from numba.cuda.tests.support import captured_stdout
import numpy as np
import cupy as cp 


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestCpuGpuCompat(CUDATestCase):
    """
    Test compatibility of CPU and GPU functions
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    @skip_on_standalone_numba_cuda
    def test_ex_cpu_gpu_compat(self):
        # ex_cpu_gpu_compat.import.begin
        from math import pi

        import numba
        from numba import cuda
        # ex_cpu_gpu_compat.import.end

        # ex_cpu_gpu_compat.allocate.begin
        X = cp.asarray([1, 10, 234])
        Y = cp.asarray([2, 2, 4014])
        Z = cp.asarray([3, 14, 2211])
        results = cp.asarray([0.0, 0.0, 0.0])
        # ex_cpu_gpu_compat.allocate.end

        # ex_cpu_gpu_compat.define.begin
        @numba.jit
        def business_logic(x, y, z):
            return 4 * z * (2 * x - (4 * y) / 2 * pi)

        # ex_cpu_gpu_compat.define.end

        # ex_cpu_gpu_compat.cpurun.begin
        print(business_logic(1, 2, 3))  # -126.79644737231007
        # ex_cpu_gpu_compat.cpurun.end

        # ex_cpu_gpu_compat.usegpu.begin
        @cuda.jit
        def f(res, xarr, yarr, zarr):
            tid = cuda.grid(1)
            if tid < len(xarr):
                # The function decorated with numba.jit may be directly reused
                res[tid] = business_logic(xarr[tid], yarr[tid], zarr[tid])

        # ex_cpu_gpu_compat.usegpu.end

        # ex_cpu_gpu_compat.launch.begin
        f.forall(len(X))(results, X, Y, Z)
        print(results)
        # [-126.79644737231007, 416.28324559588634, -218912930.2987788]
        # ex_cpu_gpu_compat.launch.end

        expect = [business_logic(x, y, z) for x, y, z in zip(X.get(), Y.get(), Z.get())]

        np.testing.assert_equal(expect, results.get())


if __name__ == "__main__":
    unittest.main()
