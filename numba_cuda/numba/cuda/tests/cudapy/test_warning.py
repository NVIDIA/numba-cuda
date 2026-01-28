# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (
    unittest,
    CUDATestCase,
    skip_on_cudasim,
)
from numba.cuda.tests.support import (
    linux_only,
    override_config,
    run_in_subprocess,
)
from numba.cuda.core.errors import (
    NumbaPerformanceWarning,
    NumbaInvalidConfigWarning,
)
from numba.cuda.core import config
import warnings
from numba.cuda.testing import DeprecatedDeviceArrayApiWarning


@skip_on_cudasim("cudasim does not raise performance warnings")
class TestWarnings(CUDATestCase):
    def test_float16_warn_if_lto_missing(self):
        fp16_kernel_invocation = """
import math
from numba import cuda

@cuda.jit
def kernel():
    x = cuda.types.float16(1.0)
    y = math.sin(x)

kernel[1,1]()
kernel[1,1]()
"""
        performance_warning = "float16 relies on LTO for performance"
        expected_warning_count = 0 if driver._have_nvjitlink() else 1
        _, err = run_in_subprocess(fp16_kernel_invocation)
        self.assertEqual(
            err.decode().count(performance_warning), expected_warning_count
        )

    def test_inefficient_launch_configuration(self):
        @cuda.jit
        def kernel():
            pass

        with override_config("CUDA_LOW_OCCUPANCY_WARNINGS", 1):
            with pytest.warns(
                NumbaPerformanceWarning, match="Grid size .+ low occupancy"
            ):
                func = kernel[1, 1]
        func()

    def test_efficient_launch_configuration(self):
        @cuda.jit
        def kernel():
            pass

        with override_config("CUDA_LOW_OCCUPANCY_WARNINGS", 1):
            with warnings.catch_warnings(record=True) as w:
                kernel[256, 256]()

        self.assertEqual(len(w), 0)

    def test_warn_on_host_array(self):
        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        N = 10
        arr_f32 = np.zeros(N, dtype=np.float32)
        func = foo[1, N]
        with override_config("CUDA_WARN_ON_IMPLICIT_COPY", 1):
            with pytest.warns(
                NumbaPerformanceWarning,
                match="Host array used in CUDA kernel will incur.+copy overhead",
            ):
                func(arr_f32, N)

    def test_pinned_warn_on_host_array(self):
        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        N = 10
        with pytest.warns(DeprecatedDeviceArrayApiWarning):
            ary = cuda.pinned_array(N, dtype=np.float32)

        func = foo[1, N]
        with override_config("CUDA_WARN_ON_IMPLICIT_COPY", 1):
            with pytest.warns(
                NumbaPerformanceWarning,
                match="Host array used in CUDA kernel will incur.+copy overhead",
            ):
                func(ary, N)

    def test_nowarn_on_mapped_array(self):
        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        N = 10
        with pytest.warns(DeprecatedDeviceArrayApiWarning):
            ary = cuda.mapped_array(N, dtype=np.float32)

        with override_config("CUDA_WARN_ON_IMPLICIT_COPY", 1):
            with warnings.catch_warnings(record=True) as w:
                foo[1, N](ary, N)

        self.assertEqual(len(w), 0)

    @linux_only
    def test_nowarn_on_managed_array(self):
        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        N = 10
        with pytest.warns(DeprecatedDeviceArrayApiWarning):
            ary = cuda.managed_array(N, dtype=np.float32)

        with override_config("CUDA_WARN_ON_IMPLICIT_COPY", 1):
            with warnings.catch_warnings(record=True) as w:
                foo[1, N](ary, N)

        self.assertEqual(len(w), 0)

    def test_nowarn_on_device_array(self):
        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        N = 10

        with pytest.warns(DeprecatedDeviceArrayApiWarning):
            ary = cuda.device_array(N, dtype=np.float32)

        with override_config("CUDA_WARN_ON_IMPLICIT_COPY", 1):
            with warnings.catch_warnings(record=True) as w:
                foo[1, N](ary, N)

        self.assertEqual(len(w), 0)

    def test_warn_on_debug_and_opt(self):
        with pytest.warns(
            NumbaInvalidConfigWarning, match="not supported by CUDA"
        ):
            cuda.jit(debug=True, opt=True)

    def test_warn_on_debug_and_opt_default(self):
        with pytest.warns(
            NumbaInvalidConfigWarning, match="not supported by CUDA"
        ):
            cuda.jit(debug=True)

    def test_no_warn_on_debug_and_no_opt(self):
        with warnings.catch_warnings(record=True) as w:
            cuda.jit(debug=True, opt=False)

        self.assertEqual(len(w), 0)

    def test_no_warn_with_no_debug_and_opt_kwargs(self):
        with warnings.catch_warnings(record=True) as w:
            cuda.jit()

        self.assertEqual(len(w), 0)

    def test_no_warn_on_debug_and_opt_with_config(self):
        with override_config("CUDA_DEBUGINFO_DEFAULT", 1):
            with override_config("OPT", config._OptLevel(0)):
                with warnings.catch_warnings(record=True) as w:
                    cuda.jit()

            self.assertEqual(len(w), 0)

            with warnings.catch_warnings(record=True) as w:
                cuda.jit(opt=False)

            self.assertEqual(len(w), 0)

        with override_config("OPT", config._OptLevel(0)):
            with warnings.catch_warnings(record=True) as w:
                cuda.jit(debug=True)

            self.assertEqual(len(w), 0)

    def test_warn_on_debug_and_opt_with_config(self):
        with override_config("CUDA_DEBUGINFO_DEFAULT", 1):
            for opt in (1, 2, 3, "max"):
                with override_config("OPT", config._OptLevel(opt)):
                    with pytest.warns(
                        NumbaInvalidConfigWarning, match="not supported by CUDA"
                    ):
                        cuda.jit()

        for opt in (1, 2, 3, "max"):
            with override_config("OPT", config._OptLevel(opt)):
                with pytest.warns(
                    NumbaInvalidConfigWarning, match="not supported by CUDA"
                ):
                    cuda.jit(debug=True)


if __name__ == "__main__":
    unittest.main()
