# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
from numba.cuda import get_current_device
from numba.cuda.cudadrv.driver import _Linker, _have_nvjitlink

from numba import cuda
from numba.cuda import config

import os
import io
import contextlib


TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")
if TEST_BIN_DIR:
    test_device_functions_a = os.path.join(
        TEST_BIN_DIR, "test_device_functions.a"
    )
    test_device_functions_cubin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cubin"
    )
    test_device_functions_cu = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cu"
    )
    test_device_functions_fatbin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.fatbin"
    )
    test_device_functions_fatbin_multi = os.path.join(
        TEST_BIN_DIR, "test_device_functions_multi.fatbin"
    )
    test_device_functions_o = os.path.join(
        TEST_BIN_DIR, "test_device_functions.o"
    )
    test_device_functions_ptx = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ptx"
    )
    test_device_functions_ltoir = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ltoir"
    )

    require_cuobjdump = (
        test_device_functions_fatbin_multi,
        test_device_functions_fatbin,
        test_device_functions_o,
    )


@unittest.skipIf(
    not TEST_BIN_DIR or not _have_nvjitlink(),
    "nvJitLink not installed or new enough (>12.3)",
)
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinker(CUDATestCase):
    def test_nvjitlink_add_file_guess_ext_linkable_code(self):
        files = (
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_cu,
            test_device_functions_fatbin,
            test_device_functions_o,
            test_device_functions_ptx,
        )
        for file in files:
            with self.subTest(file=file):
                linker = _Linker(cc=get_current_device().compute_capability)
                linker.add_file_guess_ext(file)

    def test_nvjitlink_test_add_file_guess_ext_invalid_input(self):
        with open(test_device_functions_cubin, "rb") as f:
            content = f.read()

        linker = _Linker(cc=get_current_device().compute_capability)
        with self.assertRaisesRegex(
            TypeError, "Expected path to file or a LinkableCode"
        ):
            # Feeding raw data as bytes to add_file_guess_ext should raise,
            # because there's no way to know what kind of file to treat it as
            linker.add_file_guess_ext(content)

    def test_nvjitlink_jit_with_linkable_code(self):
        files = (
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_cu,
            test_device_functions_fatbin,
            test_device_functions_o,
            test_device_functions_ptx,
        )
        for lto in [True, False]:
            for file in files:
                with self.subTest(file=file):
                    sig = "uint32(uint32, uint32)"
                    add_from_numba = cuda.declare_device("add_from_numba", sig)

                    @cuda.jit(link=[file], lto=lto)
                    def kernel(result):
                        result[0] = add_from_numba(1, 2)

                    result = cuda.device_array(1)
                    kernel[1, 1](result)
                    assert result[0] == 3

    def test_nvjitlink_jit_with_invalid_linkable_code(self):
        with open(test_device_functions_cubin, "rb") as f:
            content = f.read()
        with self.assertRaisesRegex(
            TypeError, "Expected path to file or a LinkableCode"
        ):

            @cuda.jit("void()", link=[content])
            def kernel():
                pass


@unittest.skipIf(
    not TEST_BIN_DIR or not _have_nvjitlink(),
    "nvJitLink not installed or new enough (>12.3)",
)
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinkerDumpAssembly(CUDATestCase):
    def setUp(self):
        super().setUp()
        self._prev_dump_assembly = config.DUMP_ASSEMBLY
        config.DUMP_ASSEMBLY = True

    def tearDown(self):
        config.DUMP_ASSEMBLY = self._prev_dump_assembly
        super().tearDown()

    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly(self):
        files = (
            test_device_functions_cu,
            test_device_functions_ltoir,
            test_device_functions_fatbin_multi,
        )

        for file in files:
            with self.subTest(file=file):
                if (
                    file in require_cuobjdump
                    and os.getenv("NUMBA_CUDA_TEST_WHEEL_ONLY") is not None
                ):
                    self.skipTest(
                        "wheel-only environments do not have cuobjdump"
                    )

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    sig = "uint32(uint32, uint32)"
                    add_from_numba = cuda.declare_device("add_from_numba", sig)

                    @cuda.jit(link=[file], lto=True)
                    def kernel(result):
                        result[0] = add_from_numba(1, 2)

                    result = cuda.device_array(1)
                    kernel[1, 1](result)
                    assert result[0] == 3

                self.assertTrue("ASSEMBLY (AFTER LTO)" in f.getvalue())

    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly_warn(self):
        files = (
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_fatbin,
            test_device_functions_o,
            test_device_functions_ptx,
        )

        for file in files:
            with self.subTest(file=file):
                if (
                    file in require_cuobjdump
                    and os.getenv("NUMBA_CUDA_TEST_WHEEL_ONLY") is not None
                ):
                    self.skipTest(
                        "wheel-only environments do not have cuobjdump"
                    )

                sig = "uint32(uint32, uint32)"
                add_from_numba = cuda.declare_device("add_from_numba", sig)

                @cuda.jit(link=[file], lto=True)
                def kernel(result):
                    result[0] = add_from_numba(1, 2)

                result = cuda.device_array(1)
                func = kernel[1, 1]
                with pytest.warns(
                    UserWarning,
                    match="it is not optimizable at link time, and `ignore_nonlto == True`",
                ):
                    func(result)
                assert result[0] == 3


if __name__ == "__main__":
    unittest.main()
