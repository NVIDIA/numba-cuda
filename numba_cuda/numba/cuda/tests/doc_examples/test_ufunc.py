# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Contents in this file are referenced from the sphinx-generated docs.
# "ex_cuda_ufunc" is used for markers as beginning and ending of example text.

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
def test_ufunc():
    """
    Test calling a UFunc
    """
    with captured_stdout():
        # ex_cuda_ufunc.begin
        import numpy as np
        from numba import cuda

        # A kernel calling a ufunc (sin, in this case)
        @cuda.jit
        def f(r, x):
            # Compute sin(x) with result written to r
            np.sin(x, r)

        # Declare input and output arrays
        x = np.arange(10, dtype=np.float32) - 5
        r = np.zeros_like(x)

        # Launch kernel that calls the ufunc
        f[1, 1](r, x)

        # A quick sanity check demonstrating equality of the sine computed by
        # the sin ufunc inside the kernel, and NumPy's sin ufunc
        np.testing.assert_allclose(r, np.sin(x))
        # ex_cuda_ufunc.end
