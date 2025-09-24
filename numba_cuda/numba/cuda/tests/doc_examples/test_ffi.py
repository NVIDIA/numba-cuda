# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.

import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.support import skip_unless_cffi, override_config


@skip_unless_cffi
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestFFI(CUDATestCase):
    def test_ex_linking_cu(self):
        # magictoken.ex_linking_cu.begin
        from numba import cuda
        import numpy as np
        import os

        # Path to the source containing the foreign function
        # (here assumed to be in a subdirectory called "ffi")
        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, "ffi", "functions.cu")

        # Declaration of the foreign function
        mul = cuda.declare_device(
            "mul_f32_f32", "float32(float32, float32)", link=functions_cu
        )

        # A kernel that calls mul; functions.cu is linked automatically due to
        # the call to mul.
        @cuda.jit
        def multiply_vectors(r, x, y):
            i = cuda.grid(1)

            if i < len(r):
                r[i] = mul(x[i], y[i])

        # Generate random data
        N = 32
        np.random.seed(1)
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        r = np.zeros_like(x)

        # Run the kernel
        multiply_vectors[1, 32](r, x, y)

        # Sanity check - ensure the results match those expected
        np.testing.assert_array_equal(r, x * y)
        # magictoken.ex_linking_cu.end

    def test_ex_from_buffer(self):
        from numba import cuda
        import os

        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, "ffi", "functions.cu")

        # magictoken.ex_from_buffer_decl.begin
        signature = "float32(CPointer(float32), int32)"
        sum_reduce = cuda.declare_device(
            "sum_reduce", signature, link=functions_cu
        )
        # magictoken.ex_from_buffer_decl.end

        # magictoken.ex_from_buffer_kernel.begin
        import cffi

        ffi = cffi.FFI()

        @cuda.jit
        def reduction_caller(result, array):
            array_ptr = ffi.from_buffer(array)
            result[()] = sum_reduce(array_ptr, len(array))

        # magictoken.ex_from_buffer_kernel.end

        import numpy as np

        x = np.arange(10).astype(np.float32)
        r = np.ndarray((), dtype=np.float32)

        reduction_caller[1, 1](r, x)

        expected = np.sum(x)
        actual = r[()]
        np.testing.assert_allclose(expected, actual)

    def test_ex_extra_includes(self):
        import numpy as np
        from numba import cuda
        from numba.cuda import config
        import os

        basedir = os.path.dirname(os.path.abspath(__file__))
        mul_dir = os.path.join(basedir, "ffi", "include")
        saxpy_cu = os.path.join(basedir, "ffi", "saxpy.cu")

        testdir = os.path.dirname(basedir)
        add_dir = os.path.join(testdir, "data", "include")

        includedir = ":".join([mul_dir, add_dir])
        with override_config("CUDA_NVRTC_EXTRA_SEARCH_PATHS", includedir):
            # magictoken.ex_extra_search_paths.begin
            from numba.cuda import config

            includedir = ":".join([mul_dir, add_dir])
            config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = includedir
            # magictoken.ex_extra_search_paths.end

            # magictoken.ex_extra_search_paths_kernel.begin
            sig = "float32(float32, float32, float32)"
            saxpy = cuda.declare_device("saxpy", sig=sig, link=saxpy_cu)

            @cuda.jit
            def vector_saxpy(a, x, y, res):
                i = cuda.grid(1)
                if i < len(res):
                    res[i] = saxpy(a, x[i], y[i])

            # magictoken.ex_extra_search_paths_kernel.end

            size = 10_000
            a = 3.0
            X = np.ones((size,), dtype="float32")
            Y = np.ones((size,), dtype="float32")
            R = np.zeros((size,), dtype="float32")

            block_size = 32
            num_blocks = (size // block_size) + 1

            vector_saxpy[num_blocks, block_size](a, X, Y, R)

            expected = a * X + Y
            np.testing.assert_equal(R, expected)


if __name__ == "__main__":
    unittest.main()
