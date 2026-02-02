# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_if_cupy_unavailable
import unittest
from numba.cuda import config

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    try:
        import cupy as cp
    except ImportError:
        cp = None


def reinterpret_array_type(byte_arr, start, stop, output):
    # Tested with just one thread
    val = byte_arr[start:stop].view(np.int32)[0]
    output[0] = val


class TestCudaArrayMethods(CUDATestCase):
    def setUp(self):
        self.old_nrt_setting = config.CUDA_ENABLE_NRT
        config.CUDA_ENABLE_NRT = True
        super().setUp()

    def tearDown(self):
        config.CUDA_ENABLE_NRT = self.old_nrt_setting
        super().tearDown()

    def test_reinterpret_array_type(self):
        """
        Reinterpret byte array as int32 in the GPU.
        """
        pyfunc = reinterpret_array_type
        kernel = cuda.jit(pyfunc)

        byte_arr = np.arange(256, dtype=np.uint8)
        itemsize = np.dtype(np.int32).itemsize
        for start in range(0, 256, itemsize):
            stop = start + itemsize
            expect = byte_arr[start:stop].view(np.int32)[0]

            output = np.zeros(1, dtype=np.int32)
            kernel[1, 1](byte_arr, start, stop, output)

            got = output[0]
            self.assertEqual(expect, got)

    @skip_if_cupy_unavailable
    def test_array_copy(self):
        val = np.array([1, 2, 3])[::-1]

        @cuda.jit
        def kernel(out):
            q = val.copy()
            for i in range(len(out)):
                out[i] = q[i]

        out = cp.asarray(np.zeros(len(val), dtype="float64"))

        kernel[1, 1](out)
        for i, j in zip(out.get() if not config.ENABLE_CUDASIM else out, val):
            self.assertEqual(i, j)


if __name__ == "__main__":
    unittest.main()
