# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda import HAS_NUMBA
from numba.cuda.testing import unittest, CUDATestCase

if HAS_NUMBA:
    from numba.extending import overload


@unittest.skipUnless(HAS_NUMBA, "Tests interoperability with Numba")
class TestNumbaInterop(CUDATestCase):
    def test_overload_inline_always(self):
        # From Issue #624
        def get_42():
            raise NotImplementedError()

        @overload(get_42, target="cuda", inline="always")
        def ol_blas_get_accumulator():
            def impl():
                return 42

            return impl

        @cuda.jit
        def kernel(a):
            a[0] = get_42()

        a = np.empty(1, dtype=np.float32)
        kernel[1, 1](a)
        np.testing.assert_equal(a[0], 42)
