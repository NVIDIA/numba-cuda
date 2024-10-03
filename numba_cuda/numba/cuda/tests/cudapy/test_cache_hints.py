from numba import cuda, typeof
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np

types = (
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
)

complex_types = (
    np.complex64, np.complex128
)

load_operators = (
    (cuda.ldca, 'ca'),
    (cuda.ldcg, 'cg'),
    (cuda.ldcs, 'cs'),
    (cuda.ldlu, 'lu'),
    (cuda.ldcv, 'cv')
)
store_operators = (
    (cuda.stcg, 'cg'),
    (cuda.stcs, 'cs'),
    (cuda.stwb, 'wb'),
    (cuda.stwt, 'wt')
)


class TestCacheHints(CUDATestCase):
    def test_loads(self):
        for operator, modifier in load_operators:
            @cuda.jit
            def f(r, x):
                for i in range(len(r)):
                    r[i] = operator(x, i)

            for ty in types:
                with self.subTest(operator=operator, ty=ty):
                    x = np.arange(5).astype(ty)
                    r = np.zeros_like(x)

                    f[1, 1](r, x)
                    np.testing.assert_equal(r, x)

                    # Check PTX contains a cache-policy load instruction
                    numba_type = typeof(x)
                    bitwidth = numba_type.dtype.bitwidth
                    sig = (numba_type, numba_type)
                    ptx, _ = cuda.compile_ptx(f, sig)

                    self.assertIn(f"ld.global.{modifier}.b{bitwidth}", ptx)

    def test_stores(self):
        for operator, modifier in store_operators:
            @cuda.jit
            def f(r, x):
                for i in range(len(r)):
                    operator(r, i, x[i])

            for ty in types:
                with self.subTest(operator=operator, ty=ty):
                    x = np.arange(5).astype(ty)
                    r = np.zeros_like(x)

                    f[1, 1](r, x)
                    np.testing.assert_equal(r, x)

                    # Check PTX contains a cache-policy store instruction
                    numba_type = typeof(x)
                    bitwidth = numba_type.dtype.bitwidth
                    sig = (numba_type, numba_type)
                    ptx, _ = cuda.compile_ptx(f, sig)

                    self.assertIn(f"st.global.{modifier}.b{bitwidth}", ptx)


if __name__ == '__main__':
    unittest.main()
