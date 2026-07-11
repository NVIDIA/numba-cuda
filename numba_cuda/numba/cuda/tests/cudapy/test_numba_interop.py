# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

from numba import cuda
from numba.cuda import HAS_NUMBA
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim

if HAS_NUMBA:
    from numba.extending import overload

    # User-facing repro shape from Issue #718: global overload + global kernel.
    def issue_718_get_42():
        raise NotImplementedError()

    @overload(issue_718_get_42, target="cuda", inline="always")
    def issue_718_overload_get_42():
        def impl():
            a = cuda.local.array(1, dtype=np.float32)
            a[0] = 42.0
            return a[0]

        return impl

    @cuda.jit
    def issue_718_kernel(a):
        a[0] = issue_718_get_42()


@skip_on_cudasim("Simulator does not support the extension API")
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

    def test_overload_inline_always_local_array(self):
        # From Issue #718
        # Keep the test body as close as possible to end-user kernel launch.
        a = np.empty(1, dtype=np.float32)
        d_a = cuda.to_device(a)
        issue_718_kernel[1, 1](d_a)
        d_a.copy_to_host(a)
        np.testing.assert_equal(a[0], 42.0)
