# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestCPointer(CUDATestCase):
    """
    Test simple vector addition
    """

    def setUp(self):
        # Prevent output from this test showing
        # up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_cpointer(self):
        # ex_cpointer.sig.begin
        import numpy as np
        from numba import cuda
        from numba.cuda import types

        # The first kernel argument is a pointer to a uint8 array.
        # The second argument holds the length as a uint32.
        # The return type of a kernel is always void.
        sig = types.void(types.CPointer(types.uint8), types.uint32)
        # ex_cpointer.sig.end

        # ex_cpointer.kernel.begin
        @cuda.jit(sig)
        def add_one(x, n):
            i = cuda.grid(1)
            if i < n:
                x[i] += 1

        # ex_cpointer.kernel.end

        # ex_cpointer.launch.begin
        x = cuda.to_device(np.arange(10, dtype=np.uint8))

        # Print initial values of x
        print(x.copy_to_host())  # [0 1 2 3 4 5 6 7 8 9]

        # Obtain a pointer to the data from from the CUDA Array Interface
        x_ptr = x.__cuda_array_interface__["data"][0]
        x_len = len(x)

        # Launch the kernel with the pointer and length
        add_one[1, 32](x_ptr, x_len)

        # Demonstrate that the data was updated by the kernel
        print(x.copy_to_host())  # [ 1  2  3  4  5  6  7  8  9 10]
        # ex_cpointer.launch.end


if __name__ == "__main__":
    unittest.main()
