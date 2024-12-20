
import gc
import numpy as np
import unittest
from numba.tests.support import override_config
from numba.cuda.runtime import rtsys
from numba.cuda.tests.support import EnableNRTStatsMixin
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.nrt.mock_numpy import cuda_empty, cuda_empty_like

from numba import cuda


class TestNrtRefCt(EnableNRTStatsMixin, CUDATestCase):

    def setUp(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        super(TestNrtRefCt, self).setUp()

    def tearDown(self):
        super(TestNrtRefCt, self).tearDown()

    def run(self, result=None):
        with override_config("CUDA_ENABLE_NRT", True):
            super(TestNrtRefCt, self).run(result)

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
        kernel[1, 1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)

    def test_escaping_var_init_in_loop(self):
        """
        Test issue #1297
        """

        @cuda.jit
        def g(n):

            x = cuda_empty((n, 2), np.float64)

            for i in range(n):
                y = x[i]

            for i in range(n):
                y = x[i] # noqa: F841

            return None

        init_stats = rtsys.get_allocation_stats()
        g[1, 1](10)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, 1)
        self.assertEqual(cur_stats.free - init_stats.free, 1)

    def test_invalid_computation_of_lifetime(self):
        """
        Test issue #1573
        """
        @cuda.jit
        def if_with_allocation_and_initialization(arr1, test1):
            tmp_arr = cuda_empty_like(arr1)

            for i in range(tmp_arr.shape[0]):
                pass

            if test1:
                cuda_empty_like(arr1)

        arr = np.random.random((5, 5))  # the values are not consumed

        init_stats = rtsys.get_allocation_stats()
        if_with_allocation_and_initialization[1, 1](arr, False)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc,
                         cur_stats.free - init_stats.free)

    def test_del_at_beginning_of_loop(self):
        """
        Test issue #1734
        """
        @cuda.jit
        def f(arr):
            res = 0

            for i in (0, 1):
                # `del t` is issued here before defining t.  It must be
                # correctly handled by the lowering phase.
                t = arr[i]
                if t[i] > 1:
                    res += t[i]

        arr = np.ones((2, 2))

        init_stats = rtsys.get_allocation_stats()
        f[1, 1](arr)
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc,
                         cur_stats.free - init_stats.free)


if __name__ == '__main__':
    unittest.main()
