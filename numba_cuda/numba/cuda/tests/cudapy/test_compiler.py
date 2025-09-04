# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from math import sqrt
from numba import cuda, float32, int16, int32, int64, types, uint32, void
from numba.cuda import (
    compile,
    compile_for_current_device,
    compile_ptx,
    compile_ptx_for_current_device,
)
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase


# A test function at the module scope to ensure we get the name right for the C
# ABI whether a function is at module or local scope.
def f_module(x, y):
    return x + y


@skip_on_cudasim("Compilation unsupported in the simulator")
class TestCompile(unittest.TestCase):
    def test_global_kernel(self):
        def f(r, x, y):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = x[i] + y[i]

        args = (float32[:], float32[:], float32[:])
        ptx, resty = compile_ptx(f, args)

        # Kernels should not have a func_retval parameter
        self.assertNotIn("func_retval", ptx)
        # .visible .func is used to denote a device function
        self.assertNotIn(".visible .func", ptx)
        # .visible .entry would denote the presence of a global function
        self.assertIn(".visible .entry", ptx)
        # Return type for kernels should always be void
        self.assertEqual(resty, void)

    def test_device_function(self):
        def add(x, y):
            return x + y

        args = (float32, float32)
        ptx, resty = compile_ptx(add, args, device=True)

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
        ptx, resty = compile_ptx(add, sig_int32, device=True)
        self.assertEqual(resty, int32)

        sig_int16 = int16(int16, int16)
        ptx, resty = compile_ptx(add, sig_int16, device=True)
        self.assertEqual(resty, int16)
        # Using string as signature
        sig_string = "uint32(uint32, uint32)"
        ptx, resty = compile_ptx(add, sig_string, device=True)
        self.assertEqual(resty, uint32)

    def test_fastmath(self):
        def f(x, y, z, d):
            return sqrt((x * y + z) / d)

        args = (float32, float32, float32, float32)
        ptx, resty = compile_ptx(f, args, device=True)

        # Without fastmath, fma contraction is enabled by default, but ftz and
        # approximate div / sqrt is not.
        self.assertIn("fma.rn.f32", ptx)
        self.assertIn("div.rn.f32", ptx)
        self.assertIn("sqrt.rn.f32", ptx)

        ptx, resty = compile_ptx(f, args, device=True, fastmath=True)

        # With fastmath, ftz and approximate div / sqrt are enabled
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

    def test_device_function_with_debug(self):
        # See Issue #6719 - this ensures that compilation with debug succeeds
        # with CUDA 11.2 / NVVM 7.0 onwards. Previously it failed because NVVM
        # IR version metadata was not added when compiling device functions,
        # and NVVM assumed DBG version 1.0 if not specified, which is
        # incompatible with the 3.0 IR we use. This was specified only for
        # kernels.
        def f():
            pass

        ptx, resty = compile_ptx(f, (), device=True, debug=True, opt=False)
        self.check_debug_info(ptx)

    def test_kernel_with_debug(self):
        # Inspired by (but not originally affected by) Issue #6719
        def f():
            pass

        ptx, resty = compile_ptx(f, (), debug=True, opt=False)
        self.check_debug_info(ptx)

    def check_line_info(self, ptx):
        # A .file directive should be produced and include the name of the
        # source. The path and whitespace may vary, so we accept anything
        # ending in the filename of this module.
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_line_info(self):
        def f():
            pass

        ptx, resty = compile_ptx(f, (), device=True, lineinfo=True)
        self.check_line_info(ptx)

    def test_kernel_with_line_info(self):
        def f():
            pass

        ptx, resty = compile_ptx(f, (), lineinfo=True)
        self.check_line_info(ptx)

    def test_non_void_return_type(self):
        def f(x, y):
            return x[0] + y[0]

        with self.assertRaisesRegex(TypeError, "must have void return type"):
            compile_ptx(f, (uint32[::1], uint32[::1]))

    def test_c_abi_disallowed_for_kernel(self):
        def f(x, y):
            return x + y

        with self.assertRaisesRegex(
            NotImplementedError, "The C ABI is not supported for kernels"
        ):
            compile_ptx(f, (int32, int32), abi="c")

    def test_unsupported_abi(self):
        def f(x, y):
            return x + y

        with self.assertRaisesRegex(
            NotImplementedError, "Unsupported ABI: fastcall"
        ):
            compile_ptx(f, (int32, int32), abi="fastcall")

    def test_c_abi_device_function(self):
        def f(x, y):
            return x + y

        ptx, resty = compile_ptx(f, int32(int32, int32), device=True, abi="c")
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

        # If we compile for 64-bit integers, the return type should be 64 bits
        # wide
        ptx, resty = compile_ptx(f, int64(int64, int64), device=True, abi="c")
        self.assertRegex(ptx, r"\.visible\s+\.func\s+\(\.param\s+\.b64")

    def test_c_abi_device_function_module_scope(self):
        ptx, resty = compile_ptx(
            f_module, int32(int32, int32), device=True, abi="c"
        )

        # The function name should match the Python function name, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+f_module\(",
        )

    def test_c_abi_with_abi_name(self):
        abi_info = {"abi_name": "_Z4funcii"}
        ptx, resty = compile_ptx(
            f_module,
            int32(int32, int32),
            device=True,
            abi="c",
            abi_info=abi_info,
        )

        # The function name should match the one given in the ABI info, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+_Z4funcii\(",
        )

    def test_compile_defaults_to_c_abi(self):
        ptx, resty = compile(f_module, int32(int32, int32), device=True)

        # The function name should match the Python function name, and its
        # return value should be 32 bits
        self.assertRegex(
            ptx,
            r"\.visible\s+\.func\s+\(\.param\s+\.b32\s+"
            r"func_retval0\)\s+f_module\(",
        )

    def test_compile_to_ltoir(self):
        ltoir, resty = compile(
            f_module, int32(int32, int32), device=True, output="ltoir"
        )

        # There are no tools to interpret the LTOIR output, but we can check
        # that we appear to have obtained an LTOIR file. This magic number is
        # not documented, but is expected to remain consistent.
        LTOIR_MAGIC = 0x7F4E43ED
        header = int.from_bytes(ltoir[:4], byteorder="little")
        self.assertEqual(header, LTOIR_MAGIC)
        self.assertEqual(resty, int32)

    def test_compile_to_invalid_error(self):
        illegal_output = "illegal"
        msg = f"Unsupported output type: {illegal_output}"
        with self.assertRaisesRegex(NotImplementedError, msg):
            compile(
                f_module,
                int32(int32, int32),
                device=True,
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

    def test_link_all(self):
        src = cuda.CUSource("""
        extern "C"
        __device__ int add(int *ret, int *a, int *b) {
            *ret = *a + *b;
            return 0;
        }
        """)
        add = cuda.declare_device("add", "int32(int32, int32)", link=src)

        def f(z, x, y):
            z[0] = add(x, y)

        ptx, resty = compile_ptx(f, (int32[::1], int32, int32), link_all=True)


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
        self.assertRegex(ptx, r".maxntid\s+128,\s+1,\s+1")
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
