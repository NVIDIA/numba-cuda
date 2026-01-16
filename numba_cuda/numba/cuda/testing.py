# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import platform
import shutil
import pytest
from datetime import datetime
from numba.cuda.utils import PYVERSION
from numba.cuda.cuda_paths import get_conda_ctk_libdir
from numba.cuda.cudadrv import driver, devices, libs
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda import config
from numba.cuda.tests.support import TestCase
from pathlib import Path

from typing import Iterable, Union
from io import StringIO
import unittest
import numpy as np
from numba.cuda import HAS_NUMBA

if PYVERSION >= (3, 10):
    from filecheck.matcher import Matcher
    from filecheck.options import Options
    from filecheck.parser import Parser, pattern_for_opts
    from filecheck.finput import FInput

numba_cuda_dir = Path(__file__).parent
test_data_dir = numba_cuda_dir / "tests" / "data"


@pytest.mark.usefixtures("initialize_from_pytest_config")
class CUDATestCase(TestCase):
    """
    For tests that use a CUDA device.

    Methods assertFileCheckAsm and assertFileCheckLLVM will inspect a
    CUDADispatcher and assert that the compilation artifacts match the
    FileCheck checks given in the kernel's docstring.

    Method assertFileCheckMatches can be used to assert that a given string
    matches FileCheck checks, and is not specific to CUDADispatcher.
    """

    FLOAT16_RTOL = np.finfo(np.float16).eps

    def setUp(self):
        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY

        # Disable warnings about low gpu utilization in the test suite
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        # Disable warnings about host arrays in the test suite
        config.CUDA_WARN_ON_IMPLICIT_COPY = 0

    def tearDown(self):
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy

    Signature = Union[tuple[type, ...], None]

    def _getIRContents(
        self,
        ir_result: Union[dict[Signature, str], str],
        signature: Union[Signature, None] = None,
    ) -> Iterable[str]:
        if isinstance(ir_result, str):
            assert signature is None, (
                "Cannot use signature because the kernel was only compiled for one signature"
            )
            return [ir_result]

        if signature is None:
            return list(ir_result.values())

        return [ir_result[signature]]

    def assertFileCheckAsm(
        self,
        ir_producer: CUDADispatcher,
        signature: Union[tuple[type, ...], None] = None,
        check_prefixes: tuple[str] = ("ASM",),
        **extra_filecheck_options,
    ) -> None:
        """
        Assert that the assembly output of the given CUDADispatcher matches
        the FileCheck checks given in the kernel's docstring.
        """
        ir_contents = self._getIRContents(ir_producer.inspect_asm(), signature)
        assert ir_contents, "No assembly output found for the given signature."
        assert ir_producer.__doc__ is not None, (
            "Kernel docstring is required. To pass checks explicitly, use assertFileCheckMatches."
        )
        check_patterns = ir_producer.__doc__
        for ir_content in ir_contents:
            self.assertFileCheckMatches(
                ir_content,
                check_patterns=check_patterns,
                check_prefixes=check_prefixes,
                **extra_filecheck_options,
            )

    def assertFileCheckLLVM(
        self,
        ir_producer: CUDADispatcher,
        signature: Union[tuple[type, ...], None] = None,
        check_prefixes: tuple[str] = ("LLVM",),
        **extra_filecheck_options,
    ) -> None:
        """
        Assert that the LLVM IR output of the given CUDADispatcher matches
        the FileCheck checks given in the kernel's docstring.
        """
        ir_contents = self._getIRContents(ir_producer.inspect_llvm(), signature)
        assert ir_contents, "No LLVM IR output found for the given signature."
        assert ir_producer.__doc__ is not None, (
            "Kernel docstring is required. To pass checks explicitly, use assertFileCheckMatches."
        )
        check_patterns = ir_producer.__doc__
        for ir_content in ir_contents:
            assert ir_content, (
                "LLVM IR content is empty for the given signature."
            )
            self.assertFileCheckMatches(
                ir_content,
                check_patterns=check_patterns,
                check_prefixes=check_prefixes,
                **extra_filecheck_options,
            )

    def assertFileCheckMatches(
        self,
        ir_content: str,
        check_patterns: str,
        check_prefixes: tuple[str] = ("CHECK",),
        **extra_filecheck_options,
    ) -> None:
        """
        Assert that the given string matches the passed FileCheck checks.

        Args:
            ir_content: The string to check against.
            check_patterns: The FileCheck checks to use.
            check_prefixes: The prefixes to use for the FileCheck checks.
            extra_filecheck_options: Extra options to pass to FileCheck.
        """
        if PYVERSION < (3, 10):
            self.skipTest("FileCheck requires Python 3.10 or later")
        opts = Options(
            match_filename="-",
            check_prefixes=list(check_prefixes),
            **extra_filecheck_options,
        )
        input_file = FInput(fname="-", content=ir_content)
        parser = Parser(opts, StringIO(check_patterns), *pattern_for_opts(opts))
        matcher = Matcher(opts, input_file, parser)
        matcher.stderr = StringIO()
        result = matcher.run()
        if result != 0:
            if self._dump_failed_filechecks:
                dump_directory = Path(
                    datetime.now().strftime("numba-ir-%Y_%m_%d_%H_%M_%S")
                )
                if not dump_directory.exists():
                    dump_directory.mkdir(parents=True, exist_ok=True)
                base_path = self.id().replace(".", "_")
                ir_dump = dump_directory / Path(base_path).with_suffix(".ll")
                checks_dump = dump_directory / Path(base_path).with_suffix(
                    ".checks"
                )
                with (
                    open(ir_dump, "w") as ir_file,
                    open(checks_dump, "w") as checks_file,
                ):
                    _ = ir_file.write(ir_content + "\n")
                    _ = checks_file.write(check_patterns)
                    dump_instructions = f"Reproduce with:\n\nfilecheck --check-prefixes={','.join(check_prefixes)} {checks_dump} --input-file {ir_dump}"
            else:
                dump_instructions = "Rerun with --dump-failed-filechecks to generate a reproducer."

            self.fail(
                f"FileCheck failed:\n{matcher.stderr.getvalue()}\n\n"
                + dump_instructions
            )


def skip_on_cudasim(reason):
    """Skip this test if running on the CUDA simulator"""
    assert isinstance(reason, str)
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)


skip_on_standalone_numba_cuda = unittest.skipUnless(
    HAS_NUMBA, "requires base numba install"
)


def skip_unless_cudasim(reason):
    """Skip this test if running on CUDA hardware"""
    assert isinstance(reason, str)
    return unittest.skipUnless(config.ENABLE_CUDASIM, reason)


def skip_unless_conda_cudatoolkit(reason):
    """Skip test if the CUDA toolkit was not installed by Conda"""
    assert isinstance(reason, str)
    return unittest.skipUnless(get_conda_ctk_libdir() is not None, reason)


def skip_if_external_memmgr(reason):
    """Skip test if an EMM Plugin is in use"""
    assert isinstance(reason, str)
    return unittest.skipIf(config.CUDA_MEMORY_MANAGER != "default", reason)


def skip_under_cuda_memcheck(reason):
    assert isinstance(reason, str)
    return unittest.skipIf(os.environ.get("CUDA_MEMCHECK") is not None, reason)


def skip_without_nvdisasm(reason):
    assert isinstance(reason, str)
    nvdisasm_path = shutil.which("nvdisasm")
    return unittest.skipIf(nvdisasm_path is None, reason)


def skip_with_nvdisasm(reason):
    assert isinstance(reason, str)
    nvdisasm_path = shutil.which("nvdisasm")
    return unittest.skipIf(nvdisasm_path is not None, reason)


def skip_on_arm(reason):
    assert isinstance(reason, str)
    cpu = platform.processor()
    is_arm = cpu.startswith("arm") or cpu.startswith("aarch")
    return unittest.skipIf(is_arm, reason)


def skip_on_wsl2(reason):
    """Skip test when running under WSL2.

    Detection is based on the kernel release string, which typically contains
    "microsoft-standard-WSL2" on WSL2 systems.
    """
    assert isinstance(reason, str)
    rel = platform.release().lower()
    is_wsl2 = ("microsoft-standard-wsl2" in rel) or ("wsl2" in rel)
    return unittest.skipIf(is_wsl2, reason)


def skip_if_cuda_includes_missing(fn):
    # Skip when cuda.h is not available - generally this should indicate
    # whether the CUDA includes are available or not
    reason = "CUDA include dir not available on this system"
    try:
        cuda_include_path = libs.get_cuda_include_dir()
    except FileNotFoundError:
        return unittest.skip(reason)(fn)
    cuda_h = os.path.join(cuda_include_path, "cuda.h")
    cuda_h_file = os.path.exists(cuda_h) and os.path.isfile(cuda_h)
    return unittest.skipUnless(cuda_h_file, reason)(fn)


def skip_if_curand_kernel_missing(fn):
    reason = "curand_kernel.h not available on this system"
    try:
        cuda_include_path = libs.get_cuda_include_dir()
    except FileNotFoundError:
        return unittest.skip(reason)(fn)
    curand_kernel_h = os.path.join(cuda_include_path, "curand_kernel.h")
    curand_kernel_h_file = os.path.exists(curand_kernel_h) and os.path.isfile(
        curand_kernel_h
    )
    return unittest.skipUnless(curand_kernel_h_file, reason)(fn)


def cc_X_or_above(major, minor):
    if not config.ENABLE_CUDASIM:
        cc = devices.get_context().device.compute_capability
        return cc >= (major, minor)
    else:
        return True


def skip_unless_cc_50(fn):
    return unittest.skipUnless(cc_X_or_above(5, 0), "requires cc >= 5.0")(fn)


def skip_unless_cc_53(fn):
    return unittest.skipUnless(cc_X_or_above(5, 3), "requires cc >= 5.3")(fn)


def skip_unless_cc_60(fn):
    return unittest.skipUnless(cc_X_or_above(6, 0), "requires cc >= 6.0")(fn)


def skip_unless_cc_75(fn):
    return unittest.skipUnless(cc_X_or_above(7, 5), "requires cc >= 7.5")(fn)


def xfail_unless_cudasim(fn):
    if config.ENABLE_CUDASIM:
        return fn
    else:
        return unittest.expectedFailure(fn)


def cudadevrt_missing():
    if config.ENABLE_CUDASIM:
        return False
    try:
        path = libs.get_cudalib("cudadevrt", static=True)
        libs.check_static_lib(path)
    except FileNotFoundError:
        return True
    return False


def skip_if_cudadevrt_missing(fn):
    return unittest.skipIf(cudadevrt_missing(), "cudadevrt missing")(fn)


def skip_if_nvjitlink_missing(reason):
    assert isinstance(reason, str)
    return unittest.skipIf(not driver._have_nvjitlink(), reason)


def _is_nvjitlink_13_1_and_sm_120():
    """Check if nvjitlink version is 13.1 and compute capability is 120 (sm_120).

    sm_120 refers to compute capability 12.0, represented as the tuple (12, 0).
    """
    if config.ENABLE_CUDASIM:
        return False

    try:
        from cuda.bindings import nvjitlink

        nvjitlink_ver = nvjitlink.version()
        # Check if nvjitlink version is 13.1.x
        if nvjitlink_ver[0] != 13 or nvjitlink_ver[1] != 1:
            return False

        # Check if compute capability is 12.0 (sm_120)
        cc = devices.get_context().device.compute_capability
        if cc != (12, 0):
            return False

        return True
    except (ImportError, AttributeError, RuntimeError):
        # ImportError: nvjitlink not available
        # AttributeError: version() method missing
        # RuntimeError: device context issues
        return False


def skip_on_nvjitlink_13_1_sm_120(reason):
    """Skip test when nvjitlink version is 13.1 and compute capability is sm_120.

    This is used to skip tests that fail at link time with nvjitlink 13.1 on sm_120
    GPUs (e.g., tests calling sum, mean, etc. in numba kernels).
    """
    assert isinstance(reason, str)
    return unittest.skipIf(_is_nvjitlink_13_1_and_sm_120(), reason)


class ForeignArray(object):
    """
    Class for emulating an array coming from another library through the CUDA
    Array interface. This just hides a DeviceNDArray so that it doesn't look
    like a DeviceNDArray.
    """

    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__
