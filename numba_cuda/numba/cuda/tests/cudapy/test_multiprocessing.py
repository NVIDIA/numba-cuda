# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import multiprocessing as mp
import pytest
import concurrent.futures

import numpy as np

from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.cudadrv.error import CudaDriverError
import unittest


@skip_on_cudasim("disabled for cudasim")
class TestMultiprocessing(CUDATestCase):
    @unittest.skipUnless(hasattr(mp, "get_context"), "requires mp.get_context")
    @unittest.skipUnless(os.name == "posix", "requires Unix")
    def test_fork(self):
        """
        Test fork detection.
        """
        cuda.current_context()  # force cuda initialize
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork")
        ) as exe:
            future = exe.submit(cuda.to_device, np.arange(1))

        with pytest.raises(
            CudaDriverError, match="CUDA initialized before forking"
        ):
            future.result()


if __name__ == "__main__":
    unittest.main()
