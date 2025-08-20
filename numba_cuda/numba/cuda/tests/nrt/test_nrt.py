# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import os

import numpy as np
import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.support import run_in_subprocess, override_config
from numba.cuda import get_current_device
from numba.cuda.cudadrv.nvrtc import compile
from numba import config, types
from numba.core.typing import signature
from numba import cuda
from numba.cuda.typing.templates import AbstractTemplate
from numba.cuda.cudadrv.linkable_code import (
    CUSource,
    PTXSource,
    Fatbin,
    Cubin,
    Archive,
    Object,
)

TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")

if not config.ENABLE_CUDASIM:
    from numba.cuda.memory_management.nrt import rtsys, get_include
    from numba.cuda.cudadecl import registry as cuda_decl_registry
    from numba.cuda.cudaimpl import lower as cuda_lower

    def allocate_deallocate_handle():
        """
        Handle to call NRT_Allocate and NRT_Free
        """
        pass

    @cuda_decl_registry.register_global(allocate_deallocate_handle)
    class AllocateShimImpl(AbstractTemplate):
        def generic(self, args, kws):
            return signature(types.void)

    device_fun_shim = cuda.declare_device(
        "device_allocate_deallocate", types.int32()
    )

    # wrapper to turn the above into a python callable
    def call_device_fun_shim():
        return device_fun_shim()

    @cuda_lower(allocate_deallocate_handle)
    def allocate_deallocate_impl(context, builder, sig, args):
        sig_ = types.int32()
        # call the external function, passing the pointer
        result = context.compile_internal(
            builder,
            call_device_fun_shim,
            sig_,
            (),
        )

        return result

    if TEST_BIN_DIR:

        def make_linkable_code(name, kind, mode):
            path = os.path.join(TEST_BIN_DIR, name)
            with open(path, mode) as f:
                contents = f.read()
            return kind(contents, nrt=True)

        nrt_extern_a = make_linkable_code("nrt_extern.a", Archive, "rb")
        nrt_extern_cubin = make_linkable_code("nrt_extern.cubin", Cubin, "rb")
        nrt_extern_cu = make_linkable_code(
            "nrt_extern.cu",
            CUSource,
            "rb",
        )
        nrt_extern_fatbin = make_linkable_code(
            "nrt_extern.fatbin", Fatbin, "rb"
        )
        nrt_extern_fatbin_multi = make_linkable_code(
            "nrt_extern_multi.fatbin", Fatbin, "rb"
        )
        nrt_extern_o = make_linkable_code("nrt_extern.o", Object, "rb")
        nrt_extern_ptx = make_linkable_code("nrt_extern.ptx", PTXSource, "rb")


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

    @skip_on_cudasim("CUDA Simulator does not produce PTX")
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


class TestNrtLinking(CUDATestCase):
    def run(self, result=None):
        with override_config("CUDA_ENABLE_NRT", True):
            super(TestNrtLinking, self).run(result)

    @skip_on_cudasim("CUDA Simulator does not link PTX")
    def test_nrt_detect_linked_ptx_file(self):
        src = f"#include <{get_include()}/nrt.cuh>"
        src += """
                 extern "C" __device__ int device_allocate_deallocate(int* nb_retval){
                     auto ptr = NRT_Allocate(1);
                     NRT_Free(ptr);
                     return 0;
                 }
        """
        cc = get_current_device().compute_capability
        ptx, _ = compile(src, "external_nrt.cu", cc)

        @cuda.jit(
            link=[
                PTXSource(
                    ptx.code
                    if config.CUDA_USE_NVIDIA_BINDING
                    else ptx.encode(),
                    nrt=True,
                )
            ]
        )
        def kernel():
            allocate_deallocate_handle()

        kernel[1, 1]()

    @unittest.skipIf(not TEST_BIN_DIR, "necessary binaries not generated.")
    @skip_on_cudasim("CUDA Simulator does not link code")
    def test_nrt_detect_linkable_code(self):
        codes = (
            nrt_extern_a,
            nrt_extern_cubin,
            nrt_extern_cu,
            nrt_extern_fatbin,
            nrt_extern_fatbin_multi,
            nrt_extern_o,
            nrt_extern_ptx,
        )
        for code in codes:
            with self.subTest(code=code):

                @cuda.jit(link=[code])
                def kernel():
                    allocate_deallocate_handle()

                kernel[1, 1]()


@skip_on_cudasim("CUDASIM does not have NRT statistics")
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
        from numba.cuda.memory_management import rtsys
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
        from numba.cuda.memory_management import rtsys

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
