# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
from math import sqrt
from numba import cuda
from numba.cuda import float32, int16, int32, int64, types, uint32, void
from numba.cuda import (
    compile,
    compile_for_current_device,
    compile_ptx,
    compile_ptx_for_current_device,
    compile_all,
    LinkableCode,
)
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase

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


# A test function at the module scope to ensure we get the name right for the C
# ABI whether a function is at module or local scope.
def f_module(x, y):
    return x + y


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestCompile(unittest.TestCase):
    def _handle_compile_result(self, ret, compile_function):
        ptx_or_code_list, resty = ret
        if compile_function in (compile_ptx, compile):
            ptx = ptx_or_code_list
        else:
            ptx = ptx_or_code_list[0]
        return ptx, resty

    def test_global_kernel(self):
        with self.subTest("compile_ptx"):
            self._test_global_kernel(compile_ptx, {})

        with self.subTest("compile_all"):
            self._test_global_kernel(
                compile_all, {"device": False, "abi": "numba", "output": "ptx"}
            )

    def _test_global_kernel(self, compile_function, default_kwargs):
        def f(r, x, y):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = x[i] + y[i]

        args = (float32[:], float32[:], float32[:])

        ret = compile_function(f, args, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)

        # Kernels should not have a func_retval parameter
        self.assertNotIn("func_retval", ptx)
        # .visible .func is used to denote a device function
        self.assertNotIn(".visible .func", ptx)
        # .visible .entry would denote the presence of a global function
        self.assertIn(".visible .entry", ptx)
        # Return type for kernels should always be void
        self.assertEqual(resty, void)

    def test_device_function(self):
        with self.subTest("compile_ptx"):
            self._test_device_function(compile_ptx, {"device": True})

        with self.subTest("compile_all"):
            self._test_device_function(
                compile_all, {"device": True, "abi": "c", "output": "ptx"}
            )

    def _test_device_function(self, compile_function, default_kwargs):
        def add(x, y):
            return x + y

        args = (float32, float32)

        ret = compile_function(add, args, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)

        # Device functions take a func_retval parameter for storing the
        # returned value in by reference
        self.assertIn("func_retval", ptx)
        # .visible .func is used to denote a device function
        self.assertIn(".visible .func", ptx)
        # .visible .entry would denote the presence of a global function
        self.assertNotIn(".visible .entry", ptx)
        # Inferred return type as expected?
        self.assertEqual(resty, float32)

        # Check that function's output matches signature
        sig_int32 = int32(int32, int32)
        ret = compile_function(add, sig_int32, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertEqual(resty, int32)

        sig_int16 = int16(int16, int16)
        ret = compile_function(add, sig_int16, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertEqual(resty, int16)
        # Using string as signature
        sig_string = "uint32(uint32, uint32)"
        ret = compile_function(add, sig_string, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertEqual(resty, uint32)

    def test_fastmath(self):
        with self.subTest("compile_ptx"):
            self._test_fastmath(compile_ptx, {"device": True})

        with self.subTest("compile_all"):
            self._test_fastmath(compile_all, {"device": True, "output": "ptx"})

    def _test_fastmath(self, compile_function, default_kwargs):
        def f(x, y, z, d):
            return sqrt((x * y + z) / d)

        args = (float32, float32, float32, float32)

        # Without fastmath, fma contraction is enabled by default, but ftz and
        # approximate div / sqrt are not.
        ret = compile_function(f, args, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertIn("fma.rn.f32", ptx)
        self.assertIn("div.rn.f32", ptx)
        self.assertIn("sqrt.rn.f32", ptx)

        # With fastmath, ftz and approximate div / sqrt are enabled
        ret = compile_function(f, args, fastmath=True, **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertIn("fma.rn.ftz.f32", ptx)
        self.assertIn("div.approx.ftz.f32", ptx)
        self.assertIn("sqrt.approx.ftz.f32", ptx)

    def check_debug_info(self, ptx):
        # A debug_info section should exist in the PTX. Whitespace varies
        # between CUDA toolkit versions.
        self.assertRegex(ptx, "\\.section\\s+\\.debug_info")
        # A .file directive should be produced and include the name of the
        # source. The path and whitespace may vary, so we accept anything
        # ending in the filename of this module.
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    # We did test for the presence of debuginfo here, but in practice it made
    # no sense - the C ABI wrapper generates a call instruction that has
    # nothing to correlate with the DWARF, so it would confuse the debugger
    # immediately anyway. With the resolution of Issue #588 (using separate
    # translation of each IR module when debuginfo is enabled) the debuginfo
    # isn't even produced for the ABI wrapper, because there was none present
    # in that module anyway. So this test can only be expected to fail until we
    # have a proper way of generating device functions with the C ABI without
    # requiring the hack of generating a wrapper.
    @unittest.expectedFailure
    def test_device_function_with_debug(self):
        # See Issue #6719 - this ensures that compilation with debug succeeds
        # with CUDA 11.2 / NVVM 7.0 onwards. Previously it failed because NVVM
        # IR version metadata was not added when compiling device functions,
        # and NVVM assumed DBG version 1.0 if not specified, which is
        # incompatible with the 3.0 IR we use. This was specified only for
        # kernels.

        with self.subTest("compile_ptx"):
            self._test_device_function_with_debug(
                compile_ptx, {"device": True, "debug": True, "opt": False}
            )

        with self.subTest("compile_all"):
            self._test_device_function_with_debug(
                compile_all,
                {
                    "device": True,
                    "debug": True,
                    "opt": False,
                    "output": "ptx",
                },
            )

    def _test_device_function_with_debug(
        self, compile_function, default_kwargs
    ):
        def f():
            pass

        ret = compile_function(f, (), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.check_debug_info(ptx)

    def test_kernel_with_debug(self):
        # Inspired by (but not originally affected by) Issue #6719

        with self.subTest("compile_ptx"):
            self._test_kernel_with_debug(
                compile_ptx, {"debug": True, "opt": False}
            )

        with self.subTest("compile_all"):
            self._test_kernel_with_debug(
                compile_all,
                {
                    "device": False,
                    "abi": "numba",
                    "debug": True,
                    "opt": False,
                    "output": "ptx",
                },
            )

    def _test_kernel_with_debug(self, compile_function, default_kwargs):
        def f():
            pass

        ret = compile_function(f, (), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.check_debug_info(ptx)

    def check_line_info(self, ptx):
        # A .file directive should be produced and include the name of the
        # source. The path and whitespace may vary, so we accept anything
        # ending in the filename of this module.
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_line_info(self):
        with self.subTest("compile_ptx"):
            self._test_device_function_with_line_info(
                compile_ptx, {"device": True, "lineinfo": True}
            )

        with self.subTest("compile_all"):
            self._test_device_function_with_line_info(
                compile_all,
                {
                    "device": True,
                    "abi": "numba",
                    "lineinfo": True,
                    "output": "ptx",
                },
            )

    def _test_device_function_with_line_info(
        self, compile_function, default_kwargs
    ):
        def f():
            pass

        ret = compile_function(f, (), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.check_line_info(ptx)

    def test_kernel_with_line_info(self):
        with self.subTest("compile_ptx"):
            self._test_kernel_with_line_info(compile_ptx, {"lineinfo": True})

        with self.subTest("compile_all"):
            self._test_kernel_with_line_info(
                compile_all,
                {
                    "device": False,
                    "abi": "numba",
                    "lineinfo": True,
                    "output": "ptx",
                },
            )

    def _test_kernel_with_line_info(self, compile_function, default_kwargs):
        def f():
            pass

        ret = compile_function(f, (), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.check_line_info(ptx)

    def test_non_void_return_type(self):
        def f(x, y):
            return x[0] + y[0]

        with self.subTest("compile_ptx"):
            with self.assertRaisesRegex(
                TypeError, "must have void return type"
            ):
                compile_ptx(f, (uint32[::1], uint32[::1]))

        with self.subTest("compile_all"):
            with self.assertRaisesRegex(
                TypeError, "must have void return type"
            ):
                compile_all(
                    f,
                    (uint32[::1], uint32[::1]),
                    device=False,
                    abi="numba",
                    output="ptx",
                )

    def test_c_abi_disallowed_for_kernel(self):
        def f(x, y):
            return x + y

        with self.subTest("compile_ptx"):
            with self.assertRaisesRegex(
                NotImplementedError, "The C ABI is not supported for kernels"
            ):
                compile_ptx(f, (int32, int32), abi="c")

        with self.subTest("compile_all"):
            with self.assertRaisesRegex(
                NotImplementedError, "The C ABI is not supported for kernels"
            ):
                compile_all(
                    f, (int32, int32), abi="c", device=False, output="ptx"
                )

    def test_unsupported_abi(self):
        def f(x, y):
            return x + y

        with self.subTest("compile_ptx"):
            with self.assertRaisesRegex(
                NotImplementedError, "Unsupported ABI: fastcall"
            ):
                compile_ptx(f, (int32, int32), abi="fastcall")

        with self.subTest("compile_all"):
            with self.assertRaisesRegex(
                NotImplementedError, "Unsupported ABI: fastcall"
            ):
                compile_all(f, (int32, int32), abi="fastcall", output="ptx")

    def test_c_abi_device_function(self):
        with self.subTest("compile_ptx"):
            self._test_c_abi_device_function(
                compile_ptx, {"device": True, "abi": "c"}
            )

        with self.subTest("compile_all"):
            self._test_c_abi_device_function(
                compile_all, {"device": True, "abi": "c", "output": "ptx"}
            )

    def _test_c_abi_device_function(self, compile_function, default_kwargs):
        def f(x, y):
            return x + y

        # 32-bit signature
        ret = compile_function(f, int32(int32, int32), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        # There should be no more than two parameters
        self.assertNotIn(ptx, "param_2")
        # The function name should match the Python function name (not the
        # qualname, which includes additional info), and its return value
        # should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+f\(",
        )

        # 64-bit signature should produce 64-bit return parameter
        ret = compile_function(f, int64(int64, int64), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)
        self.assertRegex(ptx, r"\.visible\s+\.func\s+\(\.param\s+\.b64")

    def test_c_abi_device_function_module_scope(self):
        with self.subTest("compile_ptx"):
            self._test_c_abi_device_function_module_scope(
                compile_ptx, {"device": True, "abi": "c"}
            )

        with self.subTest("compile_all"):
            self._test_c_abi_device_function_module_scope(
                compile_all,
                {"device": True, "abi": "c", "output": "ptx"},
            )

    def _test_c_abi_device_function_module_scope(
        self, compile_function, default_kwargs
    ):
        ret = compile_function(f_module, int32(int32, int32), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)

        # The function name should match the Python function name, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+f_module\(",
        )

    def test_c_abi_with_abi_name(self):
        abi_info = {"abi_name": "_Z4funcii"}

        with self.subTest("compile_ptx"):
            self._test_c_abi_with_abi_name(
                compile_ptx,
                {"device": True, "abi": "c", "abi_info": abi_info},
            )

        with self.subTest("compile_all"):
            self._test_c_abi_with_abi_name(
                compile_all,
                {
                    "device": True,
                    "abi": "c",
                    "abi_info": abi_info,
                    "output": "ptx",
                },
            )

    def _test_c_abi_with_abi_name(self, compile_function, default_kwargs):
        ret = compile_function(f_module, int32(int32, int32), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)

        # The function name should match the one given in the ABI info, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+_Z4funcii\(",
        )

    def test_compile_defaults_to_c_abi(self):
        with self.subTest("compile"):
            self._test_compile_defaults_to_c_abi(compile, {"device": True})

        with self.subTest("compile_all"):
            self._test_compile_defaults_to_c_abi(
                compile_all,
                {"device": True, "output": "ptx"},
            )

    def _test_compile_defaults_to_c_abi(self, compile_function, default_kwargs):
        ret = compile_function(f_module, int32(int32, int32), **default_kwargs)
        ptx, resty = self._handle_compile_result(ret, compile_function)

        # The function name should match the Python function name, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+f_module\(",
        )

    def test_compile_to_ltoir(self):
        with self.subTest("compile"):
            self._test_compile_to_ltoir(
                compile, {"device": True, "output": "ltoir"}
            )

        with self.subTest("compile_all"):
            self._test_compile_to_ltoir(
                compile_all,
                {"device": True, "abi": "c", "output": "ltoir"},
            )

    def _test_compile_to_ltoir(self, compile_function, default_kwargs):
        ret = compile_function(f_module, int32(int32, int32), **default_kwargs)
        code, resty = self._handle_compile_result(ret, compile_function)

        # There are no tools to interpret the LTOIR output, but we can check
        # that we appear to have obtained an LTOIR file. This magic number is
        # not documented, but is expected to remain consistent.
        LTOIR_MAGIC = 0x7F4E43ED
        header = int.from_bytes(code[:4], byteorder="little")
        self.assertEqual(header, LTOIR_MAGIC)
        self.assertEqual(resty, int32)

    def test_compile_to_invalid_error(self):
        illegal_output = "illegal"
        msg = f"Unsupported output type: {illegal_output}"
        with self.subTest("compile"):
            with self.assertRaisesRegex(NotImplementedError, msg):
                compile(
                    f_module,
                    int32(int32, int32),
                    device=True,
                    output=illegal_output,
                )

        with self.subTest("compile_all"):
            with self.assertRaisesRegex(NotImplementedError, msg):
                compile_all(
                    f_module,
                    int32(int32, int32),
                    device=True,
                    abi="c",
                    output=illegal_output,
                )

    def test_functioncompiler_locals(self):
        # Tests against regression fixed in:
        # https://github.com/NVIDIA/numba-cuda/pull/381
        #
        # "AttributeError: '_FunctionCompiler' object has no attribute
        # 'locals'"
        cond = None

        @cuda.jit("void(float32[::1])")
        def f(b_arg):
            b_smem = cuda.shared.array(shape=(1,), dtype=float32)

            if cond:
                b_smem[0] = b_arg[0]

    @unittest.skipIf(not TEST_BIN_DIR, "necessary binaries not generated.")
    def test_compile_all_with_external_functions(self):
        for link in [
            test_device_functions_a,
            test_device_functions_cubin,
            test_device_functions_cu,
            test_device_functions_fatbin,
            test_device_functions_fatbin_multi,
            test_device_functions_o,
            test_device_functions_ptx,
            test_device_functions_ltoir,
        ]:
            with self.subTest(link=link):
                add = cuda.declare_device(
                    "add_from_numba", "uint32(uint32, uint32)", link=[link]
                )

                def f(z, x, y):
                    z[0] = add(x, y)

                code_list, resty = compile_all(
                    f, (uint32[::1], uint32, uint32), device=False, abi="numba"
                )

                assert resty == void
                assert len(code_list) == 2
                link_obj = LinkableCode.from_path(link)
                if link_obj.kind == "cu":
                    # if link is a cu file, result contains a compiled object code
                    from numba.cuda._compat import ObjectCode

                    assert isinstance(code_list[1], ObjectCode)
                else:
                    assert code_list[1].kind == link_obj.kind

    @unittest.skipIf(not TEST_BIN_DIR, "necessary binaries not generated.")
    def test_compile_all_lineinfo(self):
        add = cuda.declare_device(
            "add", "float32(float32, float32)", link=[test_device_functions_cu]
        )

        def f(z, x, y):
            z[0] = add(x, y)

        args = (float32[::1], float32, float32)
        code_list, resty = compile_all(
            f, args, lineinfo=True, output="ptx", device=False, abi="numba"
        )
        assert len(code_list) == 2

        self.assertRegex(
            str(code_list[1].code.decode()),
            r"\.file.*test_device_functions",
        )

    @unittest.skipIf(not TEST_BIN_DIR, "necessary binaries not generated.")
    def test_compile_all_debug(self):
        add = cuda.declare_device(
            "add", "float32(float32, float32)", link=[test_device_functions_cu]
        )

        def f(z, x, y):
            z[0] = add(x, y)

        args = (float32[::1], float32, float32)
        code_list, resty = compile_all(
            f,
            args,
            debug=True,
            output="ptx",
            device=False,
            abi="numba",
            opt=False,
        )
        assert len(code_list) == 2

        self.assertRegex(
            str(code_list[1].code.decode()), r"\.section\s+\.debug_info"
        )


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestCompileForCurrentDevice(CUDATestCase):
    def _check_ptx_for_current_device(self, compile_function):
        def add(x, y):
            return x + y

        args = (float32, float32)
        ptx, resty = compile_function(add, args, device=True)

        # Check we target the current device's compute capability, or the
        # closest compute capability supported by the current toolkit.
        device_cc = cuda.get_current_device().compute_capability
        cc = cuda.cudadrv.nvrtc.find_closest_arch(device_cc)
        target = f".target sm_{cc[0]}{cc[1]}"
        self.assertIn(target, ptx)

    def test_compile_ptx_for_current_device(self):
        self._check_ptx_for_current_device(compile_ptx_for_current_device)

    def test_compile_for_current_device(self):
        self._check_ptx_for_current_device(compile_for_current_device)


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestCompileOnlyTests(unittest.TestCase):
    """For tests where we can only check correctness by examining the compiler
    output rather than observing the effects of execution."""

    def test_nanosleep(self):
        def use_nanosleep(x):
            # Sleep for a constant time
            cuda.nanosleep(32)
            # Sleep for a variable time
            cuda.nanosleep(x)

        ptx, resty = compile_ptx(use_nanosleep, (uint32,))

        nanosleep_count = 0
        for line in ptx.split("\n"):
            if "nanosleep.u32" in line:
                nanosleep_count += 1

        expected = 2
        self.assertEqual(
            expected,
            nanosleep_count,
            (
                f"Got {nanosleep_count} nanosleep instructions, "
                f"expected {expected}"
            ),
        )


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestCompileWithLaunchBounds(unittest.TestCase):
    def _test_launch_bounds_common(self, launch_bounds):
        def f():
            pass

        sig = "void()"
        ptx, resty = cuda.compile_ptx(f, sig, launch_bounds=launch_bounds)
        self.assertIsInstance(resty, types.NoneType)
        # Match either `.maxntid, 128, 1, 1` or `.maxntid 128` on a line by
        # itself:
        self.assertRegex(ptx, r".maxntid\s+128(?:,\s+1,\s+1)?\s*\n")
        return ptx

    def test_launch_bounds_scalar(self):
        launch_bounds = 128
        ptx = self._test_launch_bounds_common(launch_bounds)

        self.assertNotIn(".minnctapersm", ptx)
        self.assertNotIn(".maxclusterrank", ptx)

    def test_launch_bounds_tuple(self):
        launch_bounds = (128,)
        ptx = self._test_launch_bounds_common(launch_bounds)

        self.assertNotIn(".minnctapersm", ptx)
        self.assertNotIn(".maxclusterrank", ptx)

    def test_launch_bounds_with_min_cta(self):
        launch_bounds = (128, 2)
        ptx = self._test_launch_bounds_common(launch_bounds)

        self.assertRegex(ptx, r".minnctapersm\s+2")
        self.assertNotIn(".maxclusterrank", ptx)

    def test_launch_bounds_with_max_cluster_rank(self):
        def f():
            pass

        launch_bounds = (128, 2, 4)
        cc = (9, 0)
        sig = "void()"
        ptx, resty = cuda.compile_ptx(
            f, sig, launch_bounds=launch_bounds, cc=cc
        )
        self.assertIsInstance(resty, types.NoneType)
        self.assertRegex(ptx, r".maxntid\s+128,\s+1,\s+1")

        self.assertRegex(ptx, r".minnctapersm\s+2")
        self.assertRegex(ptx, r".maxclusterrank\s+4")

    def test_too_many_launch_bounds(self):
        def f():
            pass

        sig = "void()"
        launch_bounds = (128, 2, 4, 8)

        with self.assertRaisesRegex(ValueError, "Got 4 launch bounds:"):
            cuda.compile_ptx(f, sig, launch_bounds=launch_bounds)


if __name__ == "__main__":
    unittest.main()
