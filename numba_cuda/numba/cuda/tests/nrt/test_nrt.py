import re
import os

import numpy as np
import unittest
from numba.cuda.testing import CUDATestCase

from numba.tests.support import run_in_subprocess, override_config

from numba import cuda
from numba.cuda.runtime.nrt import rtsys


class TestNrtBasic(CUDATestCase):
    def run(self, result=None):
        with override_config("CUDA_ENABLE_NRT", True):
            super(TestNrtBasic, self).run(result)

    def test_nrt_launches(self):
        @cuda.jit
        def f(x):
            return x[:5]

        @cuda.jit
        def g():
            x = np.empty(10, np.int64)
            f(x)

        g[1, 1]()
        cuda.synchronize()

    def test_nrt_ptx_contains_refcount(self):
        @cuda.jit
        def f(x):
            return x[:5]

        @cuda.jit
        def g():
            x = np.empty(10, np.int64)
            f(x)

        g[1, 1]()

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
            x = np.empty(10, np.int64)
            x[5] = 1
            y = f(x)
            out_ary[0] = y[0]

        out_ary = np.zeros(1, dtype=np.int64)

        g[1, 1](out_ary)

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
        from numba.cuda.runtime import rtsys
        import numpy as np

        @cuda.jit
        def foo():
            x = np.arange(10)[0]

        # initialize the NRT before use
        rtsys.initialize()
        assert rtsys.memsys_stats_enabled(), "Stats not enabled"
        orig_stats = rtsys.get_allocation_stats()
        foo[1, 1]()
        new_stats = rtsys.get_allocation_stats()
        total_alloc = new_stats.alloc - orig_stats.alloc
        total_free = new_stats.free - orig_stats.free
        total_mi_alloc = new_stats.mi_alloc - orig_stats.mi_alloc
        total_mi_free = new_stats.mi_free - orig_stats.mi_free

        expected = 1
        assert total_alloc == expected, \\
            f"total_alloc != expected, {total_alloc} != {expected}"
        assert total_free == expected, \\
            f"total_free != expected, {total_free} != {expected}"
        assert total_mi_alloc == expected, \\
            f"total_mi_alloc != expected, {total_mi_alloc} != {expected}"
        assert total_mi_free == expected, \\
            f"total_mi_free != expected, {total_mi_free} != {expected}"
        """

        # Check env var explicitly being set works
        env = os.environ.copy()
        env["NUMBA_CUDA_NRT_STATS"] = "1"
        env["NUMBA_CUDA_ENABLE_NRT"] = "1"
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
        env["NUMBA_CUDA_NRT_STATS"] = "0"
        self.check_env_var_off(env)

    def test_stats_env_var_default_off(self):
        # Checks that the env var not being set is the same as "off", i.e.
        # default for Numba is off.
        env = os.environ.copy()
        env.pop("NUMBA_CUDA_NRT_STATS", None)
        self.check_env_var_off(env)

    def test_stats_status_toggle(self):
        @cuda.jit
        def foo():
            tmp = np.ones(3)
            arr = np.arange(5 * tmp[0])  # noqa: F841
            return None

        with (
            override_config("CUDA_ENABLE_NRT", True),
            override_config("CUDA_NRT_STATS", True),
        ):
            # Switch on stats
            rtsys.memsys_enable_stats()
            # check the stats are on
            self.assertTrue(rtsys.memsys_stats_enabled())

            for i in range(2):
                # capture the stats state
                stats_1 = rtsys.get_allocation_stats()
                # Switch off stats
                rtsys.memsys_disable_stats()
                # check the stats are off
                self.assertFalse(rtsys.memsys_stats_enabled())
                # run something that would move the counters were they enabled
                foo[1, 1]()
                # Switch on stats
                rtsys.memsys_enable_stats()
                # check the stats are on
                self.assertTrue(rtsys.memsys_stats_enabled())
                # capture the stats state (should not have changed)
                stats_2 = rtsys.get_allocation_stats()
                # run something that will move the counters
                foo[1, 1]()
                # capture the stats state (should have changed)
                stats_3 = rtsys.get_allocation_stats()
                # check stats_1 == stats_2
                self.assertEqual(stats_1, stats_2)
                # check stats_2 < stats_3
                self.assertLess(stats_2, stats_3)

    def test_rtsys_stats_query_raises_exception_when_disabled(self):
        # Checks that the standard rtsys.get_allocation_stats() query raises
        # when stats counters are turned off.

        rtsys.memsys_disable_stats()
        self.assertFalse(rtsys.memsys_stats_enabled())

        with self.assertRaises(RuntimeError) as raises:
            rtsys.get_allocation_stats()

        self.assertIn("NRT stats are disabled.", str(raises.exception))

    def test_nrt_explicit_stats_query_raises_exception_when_disabled(self):
        # Checks the various memsys_get_stats functions raise if queried when
        # the stats counters are disabled.
        method_variations = ("alloc", "free", "mi_alloc", "mi_free")
        for meth in method_variations:
            stats_func = getattr(rtsys, f"memsys_get_stats_{meth}")
            with self.subTest(stats_func=stats_func):
                # Turn stats off
                rtsys.memsys_disable_stats()
                self.assertFalse(rtsys.memsys_stats_enabled())
                with self.assertRaises(RuntimeError) as raises:
                    stats_func()
                self.assertIn("NRT stats are disabled.", str(raises.exception))

    def test_read_one_stat(self):
        @cuda.jit
        def foo():
            tmp = np.ones(3)
            arr = np.arange(5 * tmp[0])  # noqa: F841
            return None

        with (
            override_config("CUDA_ENABLE_NRT", True),
            override_config("CUDA_NRT_STATS", True),
        ):
            # Switch on stats
            rtsys.memsys_enable_stats()

            # Launch the kernel a couple of times to increase stats
            foo[1, 1]()
            foo[1, 1]()

            # Get stats struct and individual stats
            stats = rtsys.get_allocation_stats()
            stats_alloc = rtsys.memsys_get_stats_alloc()
            stats_mi_alloc = rtsys.memsys_get_stats_mi_alloc()
            stats_free = rtsys.memsys_get_stats_free()
            stats_mi_free = rtsys.memsys_get_stats_mi_free()

            # Check individual stats match stats struct
            self.assertEqual(stats.alloc, stats_alloc)
            self.assertEqual(stats.mi_alloc, stats_mi_alloc)
            self.assertEqual(stats.free, stats_free)
            self.assertEqual(stats.mi_free, stats_mi_free)


if __name__ == "__main__":
    unittest.main()
