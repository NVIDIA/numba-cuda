import re
import os

import numpy as np
import unittest
from unittest.mock import patch
from numba.cuda.testing import CUDATestCase

from numba.cuda.tests.nrt.mock_numpy import cuda_empty
from numba.tests.support import run_in_subprocess

from numba import cuda
from numba.cuda.runtime.nrt import rtsys


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


class TestNrtStatistics(CUDATestCase):

    def setUp(self):
        self._stream = cuda.default_stream()
        # Store the current stats state
        self.__stats_state = rtsys.memsys_stats_enabled(self._stream)

    def tearDown(self):
        # Set stats state back to whatever it was before the test ran
        if self.__stats_state:
            rtsys.memsys_enable_stats(self._stream)
        else:
            rtsys.memsys_disable_stats(self._stream)

    def test_stats_env_var_explicit_on(self):
        # Checks that explicitly turning the stats on via the env var works.
        src = """if 1:
        from numba import cuda
        from unittest.mock import patch
        from numba.cuda.runtime import rtsys
        from numba.cuda.tests.nrt.mock_numpy import cuda_arange

        @cuda.jit
        def foo():
            x = cuda_arange(10)[0]

        # initialize the NRT before use
        rtsys.initialize()
        assert rtsys.memsys_stats_enabled()
        orig_stats = rtsys.get_allocation_stats()
        foo[1, 1]()
        new_stats = rtsys.get_allocation_stats()
        total_alloc = new_stats.alloc - orig_stats.alloc
        total_free = new_stats.free - orig_stats.free
        total_mi_alloc = new_stats.mi_alloc - orig_stats.mi_alloc
        total_mi_free = new_stats.mi_free - orig_stats.mi_free

        expected = 1
        assert total_alloc == expected
        assert total_free == expected
        assert total_mi_alloc == expected
        assert total_mi_free == expected
        """

        # Check env var explicitly being set works
        env = os.environ.copy()
        env['NUMBA_CUDA_NRT_STATS'] = "1"
        env['NUMBA_CUDA_ENABLE_NRT'] = "1"
        run_in_subprocess(src, env=env)

    def check_env_var_off(self, env):

        src = """if 1:
        from numba import cuda
        import numpy as np
        from numba.cuda.runtime import rtsys

        @cuda.jit
        def foo():
            arr = np.arange(10)[0]

        assert rtsys.memsys_stats_enabled() == False
        try:
            rtsys.get_allocation_stats()
        except RuntimeError as e:
            assert "NRT stats are disabled." in str(e)
        """
        run_in_subprocess(src, env=env)

    def test_stats_env_var_explicit_off(self):
        # Checks that explicitly turning the stats off via the env var works.
        env = os.environ.copy()
        env['NUMBA_CUDA_NRT_STATS'] = "0"
        self.check_env_var_off(env)

    def test_stats_env_var_default_off(self):
        # Checks that the env var not being set is the same as "off", i.e.
        # default for Numba is off.
        env = os.environ.copy()
        env.pop('NUMBA_CUDA_NRT_STATS', None)
        self.check_env_var_off(env)


if __name__ == '__main__':
    unittest.main()
