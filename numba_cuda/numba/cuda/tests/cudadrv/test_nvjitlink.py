from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
from numba.cuda.cudadrv.driver import PyNvJitLinker

import itertools
import os
from numba.cuda import get_current_device
from numba import cuda
from numba import config

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

    def test_nvjitlink_create(self):
        patched_linker = PyNvJitLinker(cc=(7, 5))
        assert "-arch=sm_75" in patched_linker.options

    def test_nvjitlink_create_no_cc_error(self):
        # nvJitLink expects at least the architecture to be specified.
        with self.assertRaisesRegex(
            RuntimeError, "PyNvJitLinker requires CC to be specified"
        ):
            PyNvJitLinker()

    def test_nvjitlink_invalid_arch_error(self):
        from pynvjitlink.api import NvJitLinkError

        # CC 0.0 is not a valid compute capability
        with self.assertRaisesRegex(
            NvJitLinkError, "NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"
        ):
            PyNvJitLinker(cc=(0, 0))

    def test_nvjitlink_invalid_cc_type_error(self):
        with self.assertRaisesRegex(
            TypeError, "`cc` must be a list or tuple of length 2"
        ):
            PyNvJitLinker(cc=0)

    def test_nvjitlink_ptx_compile_options(self):

        max_registers = (None, 32)
        lineinfo = (False, True)
        lto = (False, True)
        additional_flags = (None, ("-g",), ("-g", "-time"))
        for (
            max_registers_i,
            line_info_i,
            lto_i,
            additional_flags_i,
        ) in itertools.product(max_registers, lineinfo, lto, additional_flags):
            with self.subTest(
                max_registers=max_registers_i,
                lineinfo=line_info_i,
                lto=lto_i,
                additional_flags=additional_flags_i,
            ):
                patched_linker = PyNvJitLinker(
                    cc=(7, 5),
                    max_registers=max_registers_i,
                    lineinfo=line_info_i,
                    lto=lto_i,
                    additional_flags=additional_flags_i,
                )
                assert "-arch=sm_75" in patched_linker.options

                if max_registers_i:
                    assert (
                        f"-maxrregcount={max_registers_i}"
                        in patched_linker.options
                    )
                else:
                    assert "-maxrregcount" not in patched_linker.options

                if line_info_i:
                    assert "-lineinfo" in patched_linker.options
                else:
                    assert "-lineinfo" not in patched_linker.options

                if lto_i:
                    assert "-lto" in patched_linker.options
                else:
                    assert "-lto" not in patched_linker.options

                if additional_flags_i:
                    for flag in additional_flags_i:
                        assert flag in patched_linker.options

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
        for file in files:
            with self.subTest(file=file):
                sig = "uint32(uint32, uint32)"
                add_from_numba = cuda.declare_device("add_from_numba", sig)

                @cuda.jit(link=[file])
                def kernel(result):
                    result[0] = add_from_numba(1, 2)

                result = cuda.device_array(1)
                kernel[1, 1](result)
                assert result[0] == 3

    def test_nvjitlink_jit_with_linkable_code_lto(self):
        file = test_device_functions_ltoir

        sig = "uint32(uint32, uint32)"
        add_from_numba = cuda.declare_device("add_from_numba", sig)

        @cuda.jit(link=[file], lto=True)
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


if __name__ == "__main__":
    unittest.main()
