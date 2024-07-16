import gc
import numpy as np
import unittest
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin

from numba import cuda

class TestNrtRefCt(EnableNRTStatsMixin, TestCase):

    def setUp(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        super(TestNrtRefCt, self).setUp()

    def test_no_return(self):
        """
        Test issue #1291
        """

        @cuda.jit
        def kernel():
            for i in 10:
                temp = np.zeros(2)
            return 0

        init_stats = rtsys.get_allocation_stats()
        kernel[1,1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)

if __name__ == '__main__':
    unittest.main()
