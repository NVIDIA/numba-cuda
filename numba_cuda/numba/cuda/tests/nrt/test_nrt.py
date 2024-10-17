import gc
import numpy as np
import unittest
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
from numba.cuda.testing import CUDATestCase

from numba import cuda

from numba.core import errors, types
from numba.core.extending import overload
from numba.np.arrayobj import (_check_const_str_dtype, is_nonelike,
                               ty_parse_dtype, ty_parse_shape, numpy_empty_nd)


def cuda_empty(shape, dtype):
    pass


@overload(cuda_empty)
def ol_cuda_empty(shape, dtype):
    _check_const_str_dtype("empty", dtype)
    if (dtype is float or
        (isinstance(dtype, types.Function) and dtype.typing_key is float) or
            is_nonelike(dtype)): #default
        nb_dtype = types.double
    else:
        nb_dtype = ty_parse_dtype(dtype)

    ndim = ty_parse_shape(shape)
    if nb_dtype is not None and ndim is not None:
        retty = types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        def impl(shape, dtype):
            return numpy_empty_nd(shape, dtype, retty)
        return impl
    else:
        msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
        raise errors.TypingError(msg)


@unittest.skip
class TestNrtRefCt(EnableNRTStatsMixin, TestCase):

    def setUp(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        super(TestNrtRefCt, self).setUp()

    def test_no_return(self):
        """
        Test issue #1291
        """
        n = 10

        @cuda.jit
        def kernel():
            for i in range(n):
                temp = np.zeros(2) # noqa: F841
            return 0

        init_stats = rtsys.get_allocation_stats()
        kernel[1,1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)


class TestNrtBasic(CUDATestCase):
    def test_nrt_launches(self):
        from pynvjitlink.patch import patch_numba_linker

        patch_numba_linker()

        @cuda.jit
        def f(x):
            return x[:5]

        @cuda.jit
        def g():
            x = cuda_empty(10, np.int64)
            f(x)

        g[1,1]()
        cuda.synchronize()

    def test_nrt_returns_correct(self):
        from pynvjitlink.patch import patch_numba_linker

        patch_numba_linker()

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

        g[1,1](out_ary)

        self.assertEqual(out_ary[0], 1)


if __name__ == '__main__':
    unittest.main()
