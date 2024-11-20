import gc
import numpy as np
import unittest
from unittest.mock import patch
from numba.core.runtime import rtsys
from numba.tests.support import EnableNRTStatsMixin
from numba.cuda.testing import CUDATestCase

from .mock_numpy import cuda_empty

from numba import cuda


class TestNrtRefCt(EnableNRTStatsMixin, CUDATestCase):

    def setUp(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        super(TestNrtRefCt, self).setUp()

    @unittest.expectedFailure
    def test_no_return(self):
        """
        Test issue #1291
        """
        n = 10

        @cuda.jit
        def kernel():
            for i in range(n):
                temp = cuda_empty(2, np.float64) # noqa: F841
            return None

        init_stats = rtsys.get_allocation_stats()

        with patch('numba.config.CUDA_ENABLE_NRT', True, create=True):
            kernel[1,1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)


class TestNrtBasic(CUDATestCase):
    def test_nrt_launches(self):
        @cuda.jit
        def f(x):
            return x[:5]

        @cuda.jit
        def g():
            x = cuda_empty(10, np.int64)
            f(x)

        with patch('numba.config.CUDA_ENABLE_NRT', True, create=True):
            g[1,1]()
        cuda.synchronize()

    def test_nrt_returns_correct(self):
        @cuda.jit
        def f(x):
            return x[5:]

        @cuda.jit
        def g(out_ary):
            x = cuda_empty(10, np.int64)
            x[5] = 1
            y = f(x)
            out_ary[0] = y[0]

        out_ary = np.zeros(1, dtype=np.int64)

        with patch('numba.config.CUDA_ENABLE_NRT', True, create=True):
            g[1,1](out_ary)

        self.assertEqual(out_ary[0], 1)


if __name__ == '__main__':
    unittest.main()
