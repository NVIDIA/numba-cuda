from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
from numba.cuda.cudadrv.driver import PyNvJitLinker
from numba.cuda import get_current_device

from numba import cuda
from numba import config
from numba.tests.support import run_in_subprocess, override_config

try:
    import pynvjitlink  # noqa: F401
    PYNVJITLINK_INSTALLED = True
except ImportError:
    PYNVJITLINK_INSTALLED = False


import itertools
import os
import io
import contextlib
import warnings


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

@unittest.skipIf(
    not config.CUDA_ENABLE_PYNVJITLINK or not TEST_BIN_DIR,
    "pynvjitlink not enabled"
)
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinker(CUDATestCase):
    _NUMBA_NVIDIA_BINDING_0_ENV = {"NUMBA_CUDA_USE_NVIDIA_BINDING": "0"}



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
                patched_linker = PyNvJitLinker(
                    cc=get_current_device().compute_capability
                )
                patched_linker.add_file_guess_ext(file)

    def test_nvjitlink_test_add_file_guess_ext_invalid_input(self):
        with open(test_device_functions_cubin, "rb") as f:
            content = f.read()

        patched_linker = PyNvJitLinker(
            cc=get_current_device().compute_capability
        )
        with self.assertRaisesRegex(
            TypeError, "Expected path to file or a LinkableCode"
        ):
            # Feeding raw data as bytes to add_file_guess_ext should raise,
            # because there's no way to know what kind of file to treat it as
            patched_linker.add_file_guess_ext(content)

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

    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly(self):
        files = [
            test_device_functions_cu,
            #test_device_functions_ltoir,
            #test_device_functions_fatbin_multi
        ]

        config.DUMP_ASSEMBLY = True

        for file in files:
            with self.subTest(file=file):
                f = io.StringIO()
                #with contextlib.redirect_stdout(f):
                sig = "uint32(uint32, uint32)"
                add_from_numba = cuda.declare_device("add_from_numba", sig)

                @cuda.jit(link=[file], lto=True)
                def kernel(result):
                    result[0] = add_from_numba(1, 2)

                result = cuda.device_array(1)
                kernel[1, 1](result)
                assert result[0] == 3
            
                self.assertTrue("ASSEMBLY (AFTER LTO)" in f.getvalue())

        config.DUMP_ASSEMBLY = False

    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly_warn(self):
        files = [
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_fatbin,
            test_device_functions_o,
            test_device_functions_ptx,
        ]

        config.DUMP_ASSEMBLY = True

        for file in files:
            with self.subTest(file=file):
                with warnings.catch_warnings(record=True) as w:
                    with contextlib.redirect_stdout(None): # suppress other PTX
                        sig = "uint32(uint32, uint32)"
                        add_from_numba = cuda.declare_device(
                            "add_from_numba", sig
                        )
                        @cuda.jit(link=[file], lto=True)
                        def kernel(result):
                            result[0] = add_from_numba(1, 2)

                        result = cuda.device_array(1)
                        kernel[1, 1](result)
                        assert result[0] == 3

                assert len(w) == 1
                self.assertIn("it is not optimizable at link time, and "
                              "`ignore_nonlto == True`", str(w[0].message))

        config.DUMP_ASSEMBLY = False

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
    not PYNVJITLINK_INSTALLED, reason="Pynvjitlink is not installed"
)
class TestLinkerUsage(CUDATestCase):
    """Test that whether pynvjitlink can be enabled by both environment variable
    and modification of config at runtime.
    """

    src = """if 1:
        import os
        from numba import cuda, config

        {config}

        TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")
        if TEST_BIN_DIR:
            test_device_functions_cubin = os.path.join(
                TEST_BIN_DIR, "test_device_functions.cubin"
            )

        sig = "uint32(uint32, uint32)"
        add_from_numba = cuda.declare_device("add_from_numba", sig)

        @cuda.jit(link=[test_device_functions_cubin], lto=True)
        def kernel(result):
            result[0] = add_from_numba(1, 2)

        result = cuda.device_array(1)
        kernel[1, 1](result)
        assert result[0] == 3
        """

    def test_linker_enabled_envvar(self):
        env = os.environ.copy()
        env['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = "1"
        run_in_subprocess(self.src.format(config=""), env=env)

    def test_linker_disabled_envvar(self):
        env = os.environ.copy()
        env.pop('NUMBA_CUDA_ENABLE_PYNVJITLINK', None)
        with self.assertRaisesRegex(
            AssertionError, "LTO and additional flags require PyNvJitLinker"
        ):
            # Actual error raised is `ValueError`, but `run_in_subprocess`
            # reraises as AssertionError.
            run_in_subprocess(self.src.format(config=""), env=env)

    def test_linker_enabled_config(self):
        env = os.environ.copy()
        env.pop('NUMBA_CUDA_ENABLE_PYNVJITLINK', None)
        run_in_subprocess(self.src.format(
            config="config.CUDA_ENABLE_PYNVJITLINK = True"), env=env)

    def test_linker_disabled_config(self):
        env = os.environ.copy()
        env.pop('NUMBA_CUDA_ENABLE_PYNVJITLINK', None)
        with override_config("CUDA_ENABLE_PYNVJITLINK", False):
            with self.assertRaisesRegex(
                AssertionError, "LTO and additional flags require PyNvJitLinker"
            ):
                run_in_subprocess(self.src.format(
                    config="config.CUDA_ENABLE_PYNVJITLINK = False"), env=env)


if __name__ == "__main__":
    unittest.main()
