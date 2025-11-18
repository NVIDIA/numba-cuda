# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba.cuda import vectorize
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestVectorizeComplex(CUDATestCase):
    def test_vectorize_complex(self):
        @vectorize(["complex128(complex128)"], target="cuda")
        def vcomp(a):
            return a * a + 1.0

        A = np.arange(5, dtype=np.complex128)
        B = vcomp(A)
        self.assertTrue(np.allclose(A * A + 1.0, B))


if __name__ == "__main__":
    unittest.main()
