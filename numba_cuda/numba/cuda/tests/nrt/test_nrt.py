import re
import numpy as np
import unittest
from unittest.mock import patch
from numba.cuda.testing import CUDATestCase

from numba.cuda.tests.nrt.mock_numpy import cuda_empty

from numba import cuda


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

    def test_nrt_ptx_contains_refcount(self):
        @cuda.jit
        def f(x):
            return x[:5]

        @cuda.jit
        def g():
            x = cuda_empty(10, np.int64)
            f(x)

        with patch('numba.config.CUDA_ENABLE_NRT', True, create=True):
            g[1,1]()

        ptx = next(iter(g.inspect_asm().values()))

        # The following checks that a `call` PTX instruction is
        # emitted for NRT_MemInfo_alloc_aligned, NRT_incref and
        # NRT_decref
        p1 = r"call\.uni(.|\n)*NRT_MemInfo_alloc_aligned"
        match = re.search(p1, ptx)
        assert match is not None

        p2 = r"call\.uni.*\n.*NRT_incref"
        match = re.search(p2, ptx)
        assert match is not None

        p3 = r"call\.uni.*\n.*NRT_decref"
        match = re.search(p3, ptx)
        assert match is not None

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
