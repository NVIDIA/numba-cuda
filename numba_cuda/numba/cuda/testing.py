import os
import platform
import shutil

from numba.tests.support import SerialMixin
from numba.cuda.cuda_paths import get_conda_ctk
from numba.cuda.cudadrv import driver, devices, libs
from numba.core import config
from numba.tests.support import TestCase
from pathlib import Path
from filecheck.matcher import Matcher, Options
from filecheck.parser import Parser, pattern_for_opts
from filecheck.finput import FInput
from io import StringIO
import unittest

numba_cuda_dir = Path(__file__).parent
test_data_dir = numba_cuda_dir / "tests" / "data"


class CUDATestCase(SerialMixin, TestCase):
    """
    For tests that use a CUDA device. Test methods in a CUDATestCase must not
    be run out of module order, because the ContextResettingTestCase may reset
    the context and destroy resources used by a normal CUDATestCase if any of
    its tests are run between tests from a CUDATestCase.
    """

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


class ContextResettingTestCase(CUDATestCase):
    """
    For tests where the context needs to be reset after each test. Typically
    these inspect or modify parts of the context that would usually be expected
    to be internal implementation details (such as the state of allocations and
    deallocations, etc.).
    """

    def tearDown(self):
        super().tearDown()
        from numba.cuda.cudadrv.devices import reset

        reset()


def skip_on_cudasim(reason):
    """Skip this test if running on the CUDA simulator"""
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)


def skip_unless_cudasim(reason):
    """Skip this test if running on CUDA hardware"""
    return unittest.skipUnless(config.ENABLE_CUDASIM, reason)


def skip_unless_conda_cudatoolkit(reason):
    """Skip test if the CUDA toolkit was not installed by Conda"""
    return unittest.skipUnless(get_conda_ctk() is not None, reason)


def skip_if_external_memmgr(reason):
    """Skip test if an EMM Plugin is in use"""
    return unittest.skipIf(config.CUDA_MEMORY_MANAGER != "default", reason)


def skip_under_cuda_memcheck(reason):
    return unittest.skipIf(os.environ.get("CUDA_MEMCHECK") is not None, reason)


def skip_without_nvdisasm(reason):
    nvdisasm_path = shutil.which("nvdisasm")
    return unittest.skipIf(nvdisasm_path is None, reason)


def skip_with_nvdisasm(reason):
    nvdisasm_path = shutil.which("nvdisasm")
    return unittest.skipIf(nvdisasm_path is not None, reason)


def skip_on_arm(reason):
    cpu = platform.processor()
    is_arm = cpu.startswith("arm") or cpu.startswith("aarch")
    return unittest.skipIf(is_arm, reason)


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


def skip_if_mvc_enabled(reason):
    """Skip a test if Minor Version Compatibility is enabled"""
    return unittest.skipIf(
        config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY, reason
    )


def skip_if_mvc_libraries_unavailable(fn):
    libs_available = False
    try:
        import cubinlinker  # noqa: F401
        import ptxcompiler  # noqa: F401

        libs_available = True
    except ImportError:
        pass

    return unittest.skipUnless(
        libs_available, "Requires cubinlinker and ptxcompiler"
    )(fn)


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


def skip_with_cuda_python(reason):
    return unittest.skipIf(driver.USE_NV_BINDING, reason)


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
    return unittest.skipIf(not driver._have_nvjitlink(), reason)


class ForeignArray(object):
    """
    Class for emulating an array coming from another library through the CUDA
    Array interface. This just hides a DeviceNDArray so that it doesn't look
    like a DeviceNDArray.
    """

    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__


def filecheck_ir(
    ir_content: str,
    check_patterns: str,
    check_prefixes: list[str] = ["CHECK"],
    **extra_filecheck_options: dict[str, str | int],
) -> bool:
    """Run filecheck on IR content with check patterns. Raises an AssertionError if the checks fail."""
    opts = Options(
        match_filename="-",
        check_prefixes=check_prefixes,
        **extra_filecheck_options,
    )
    fin = FInput(fname="-", content=ir_content)
    parser = Parser(opts, StringIO(check_patterns), *pattern_for_opts(opts))
    matcher = Matcher(opts, fin, parser)
    result = matcher.run()
    if result != 0:
        raise AssertionError(
            (
                f"FileCheck failed with exit code {result}\n\n"
                f"Check prefixes:\n{check_prefixes}\n\n"
                f"Check patterns:\n{check_patterns}\n"
                f"IR:\n{ir_content}\n\n"
            )
        )
    return True


class FileCheckKernel:
    def __init__(self, kernel, check_patterns: str | None = None):
        self.kernel = kernel
        self.check_patterns = (
            check_patterns if check_patterns else kernel.__doc__
        )

    def check_llvm(
        self,
        signature: tuple[type, ...] | None = None,
        check_prefixes: list[str] = ["LLVM"],
    ) -> bool:
        llvm = self.kernel.inspect_llvm()
        if signature:
            llvm = llvm[signature]
        return filecheck_ir(
            llvm, self.check_patterns, check_prefixes=check_prefixes
        )

    def check_asm(
        self,
        signature: tuple[type, ...] | None = None,
        check_prefixes: list[str] = ["ASM"],
    ) -> bool:
        asm = self.kernel.inspect_asm()
        if signature:
            asm = asm[signature]
        return filecheck_ir(
            asm, self.check_patterns, check_prefixes=check_prefixes
        )

    def check_sass(
        self,
        signature: tuple[type, ...] | None = None,
        check_prefixes: list[str] = ["SASS"],
    ) -> bool:
        sass = self.kernel.inspect_sass()
        if signature:
            sass = sass[signature]
        return filecheck_ir(
            sass, self.check_patterns, check_prefixes=check_prefixes
        )
