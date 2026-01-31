# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from numba.cuda.testing import (
    CUDATestCase,
    skip_if_cupy_unavailable,
    skip_on_cudasim,
)
from numba.cuda.tests.support import captured_stdout
import cupy as cp


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestGlobals(CUDATestCase):
    """
    Tests demonstrating how global variables are captured in CUDA kernels.
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

    @skip_if_cupy_unavailable
    def test_ex_globals_constant_capture(self):
        """
        Test demonstrating how global variables are captured as constants.
        """
        # magictoken.ex_globals_constant_capture.begin
        import numpy as np
        from numba import cuda

        TAX_RATE = 0.08
        PRICES = np.array([10.0, 25.0, 5.0, 15.0, 30.0], dtype=np.float64)

        @cuda.jit
        def compute_totals(quantities, totals):
            i = cuda.grid(1)
            if i < totals.size:
                totals[i] = quantities[i] * PRICES[i] * (1 + TAX_RATE)

        d_quantities = cp.asarray(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        d_totals = cp.zeros(5, dtype=np.float64)

        # First kernel call - compiles and captures values
        compute_totals[1, 32](d_quantities, d_totals)
        print("Value of d_totals:", d_totals.get())

        # These modifications have no effect on subsequent kernel calls
        TAX_RATE = 0.10  # noqa: F841
        PRICES[:] = [20.0, 50.0, 10.0, 30.0, 60.0]

        # Second kernel call still uses the original values
        compute_totals[1, 32](d_quantities, d_totals)
        print("Value of d_totals:", d_totals.get())
        # magictoken.ex_globals_constant_capture.end

        # Verify the values are the same (original values were captured)
        expected = np.array([10.8, 54.0, 16.2, 64.8, 162.0])
        np.testing.assert_allclose(d_totals.get(), expected)

    @skip_if_cupy_unavailable
    def test_ex_globals_device_array_capture(self):
        """
        Test demonstrating how global device arrays are captured by pointer.
        """
        # magictoken.ex_globals_device_array_capture.begin
        import numpy as np
        from numba import cuda

        # Global device array - pointer is captured, not data
        PRICES = cp.asarray(
            np.array([10.0, 25.0, 5.0, 15.0, 30.0], dtype=np.float32)
        )

        @cuda.jit
        def compute_totals(quantities, totals):
            i = cuda.grid(1)
            if i < totals.size:
                totals[i] = quantities[i] * PRICES[i]

        d_quantities = cp.asarray(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        d_totals = cp.zeros(5, dtype=np.float32)

        # First kernel call
        compute_totals[1, 32](d_quantities, d_totals)
        print(d_totals.get())  # [10. 25.  5. 15. 30.]

        # Mutate the device array in-place
        PRICES[:] = cp.array([20.0, 50.0, 10.0, 30.0, 60.0], dtype=np.float32)

        # Second kernel call sees the updated values
        compute_totals[1, 32](d_quantities, d_totals)
        print(d_totals.get())  # [20. 50. 10. 30. 60.]
        # magictoken.ex_globals_device_array_capture.end

        # Verify the second call sees updated values
        expected = np.array([20.0, 50.0, 10.0, 30.0, 60.0], dtype=np.float32)
        np.testing.assert_allclose(d_totals.get(), expected)


if __name__ == "__main__":
    unittest.main()
