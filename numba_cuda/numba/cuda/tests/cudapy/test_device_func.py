# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import cffi

import numpy as np

from numba.cuda.testing import (
    skip_if_curand_kernel_missing,
    skip_on_cudasim,
    test_data_dir,
    unittest,
    CUDATestCase,
)
from numba import cuda
from numba.cuda import float32, int32, types
from numba.cuda.core.errors import TypingError
from numba.cuda.tests.support import skip_unless_cffi
from numba.cuda.testing import skip_on_standalone_numba_cuda
from types import ModuleType
from numba.cuda import HAS_NUMBA, config

if HAS_NUMBA:
    from numba import jit

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp


class TestDeviceFunc(CUDATestCase):
    def test_use_add2f(self):
        @cuda.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        def use_add2f(ary):
            i = cuda.grid(1)
            ary[i] = add2f(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(use_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def test_indirect_add2f(self):
        @cuda.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        @cuda.jit("float32(float32, float32)", device=True)
        def indirect(a, b):
            return add2f(a, b)

        def indirect_add2f(ary):
            i = cuda.grid(1)
            ary[i] = indirect(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(indirect_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def _check_cpu_dispatcher(self, add):
        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
            ary[i] = add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    @skip_on_standalone_numba_cuda
    def test_cpu_dispatcher(self):
        # Test correct usage
        @jit
        def add(a, b):
            return a + b

        self._check_cpu_dispatcher(add)

    @skip_on_cudasim("not supported in cudasim")
    @skip_on_standalone_numba_cuda
    def test_cpu_dispatcher_invalid(self):
        # Test invalid usage
        # Explicit signature disables compilation, which also disable
        # compiling on CUDA.
        @jit("(i4, i4)")
        def add(a, b):
            return a + b

        # Check that the right error message is provided.
        with self.assertRaises(TypingError) as raises:
            self._check_cpu_dispatcher(add)
        msg = "Untyped global name 'add':.*using cpu function on device"
        expected = re.compile(msg)
        self.assertTrue(expected.search(str(raises.exception)) is not None)

    @skip_on_standalone_numba_cuda
    def test_cpu_dispatcher_other_module(self):
        @jit
        def add(a, b):
            return a + b

        mymod = ModuleType(name="mymod")
        mymod.add = add
        del add

        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
            ary[i] = mymod.add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    @skip_on_cudasim("not supported in cudasim")
    def test_inspect_llvm(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile_device(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn("foo", fname)

        llvm = foo.inspect_llvm(args)
        # Check that the compiled function name is in the LLVM.
        self.assertIn(fname, llvm)

    @skip_on_cudasim("not supported in cudasim")
    def test_inspect_asm(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile_device(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn("foo", fname)

        ptx = foo.inspect_asm(args)
        # Check that the compiled function name is in the PTX
        self.assertIn(fname, ptx)

    @skip_on_cudasim("not supported in cudasim")
    def test_inspect_sass_disallowed(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        with self.assertRaises(RuntimeError) as raises:
            foo.inspect_sass((int32, int32))

        self.assertIn(
            "Cannot inspect SASS of a device function", str(raises.exception)
        )

    @skip_on_cudasim("cudasim will allow calling any function")
    def test_device_func_as_kernel_disallowed(self):
        @cuda.jit(device=True)
        def f():
            pass

        with self.assertRaises(RuntimeError) as raises:
            f[1, 1]()

        self.assertIn(
            "Cannot compile a device function as a kernel",
            str(raises.exception),
        )

    @skip_on_cudasim("cudasim ignores casting by jit decorator signature")
    @skip_if_cupy_unavailable
    def test_device_casting(self):
        # Ensure that casts to the correct type are forced when calling a
        # device function with a signature. This test ensures that:
        #
        # - We don't compile a new specialization of rgba for float32 when we
        #   shouldn't
        # - We insert a cast when calling rgba, as opposed to failing to type.

        @cuda.jit("int32(int32, int32, int32, int32)", device=True)
        def rgba(r, g, b, a):
            return (
                ((r & 0xFF) << 16)
                | ((g & 0xFF) << 8)
                | ((b & 0xFF) << 0)
                | ((a & 0xFF) << 24)
            )

        @cuda.jit
        def rgba_caller(x, channels):
            x[0] = rgba(channels[0], channels[1], channels[2], channels[3])

        x = cp.asarray([1], dtype=np.int32)
        channels = cp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        rgba_caller[1, 1](x, channels)

        self.assertEqual(0x04010203, x[0])


times2_cu = cuda.CUSource("""
extern "C" __device__
int times2(int *out, int a)
{
  *out = a * 2;
  return 0;
}
""")

times3_cu = cuda.CUSource("""
extern "C" __device__
int times3(int *out, int a)
{
  *out = a * 3;
  return 0;
}
""")

times4_cu = cuda.CUSource("""
extern "C" __device__
int times2(int *out, int a);

extern "C" __device__
int times4(int *out, int a)
{
  int tmp;
  times2(&tmp, a);
  *out = tmp * 2;
  return 0;
}
""")

jitlink_user_cu = cuda.CUSource("""
extern "C" __device__
int array_mutator(void *out, int *a);

extern "C" __device__
int use_array_mutator(void *out, int *a) {
  array_mutator(out, a);
  return 0;
}
""")

rng_cu = cuda.CUSource("""
#include <curand_kernel.h>

extern "C" __device__
int random_number(unsigned int *out, unsigned long long seed)
{
  // Initialize state
  curandStateXORWOW_t state;
  unsigned long long sequence = 1;
  unsigned long long offset = 0;
  curand_init(seed, sequence, offset, &state);

  // Generate one random number
  *out = curand(&state);

  // Report no exception
  return 0;
}""")


@skip_on_cudasim("External functions unsupported in the simulator")
class TestDeclareDevice(CUDATestCase):
    def check_api(self, decl):
        self.assertEqual(decl.name, "f1")
        self.assertEqual(decl.sig.args, (float32[:],))
        self.assertEqual(decl.sig.return_type, int32)

    def test_declare_device_signature(self):
        f1 = cuda.declare_device("f1", int32(float32[:]))
        self.check_api(f1)

    def test_declare_device_string(self):
        f1 = cuda.declare_device("f1", "int32(float32[:])")
        self.check_api(f1)

    def test_bad_declare_device_tuple(self):
        with self.assertRaisesRegex(TypeError, "Return type"):
            cuda.declare_device("f1", (float32[:],))

    def test_bad_declare_device_string(self):
        with self.assertRaisesRegex(TypeError, "Return type"):
            cuda.declare_device("f1", "(float32[:],)")

    def test_link_cu_source(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = times2(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)

    def _test_link_multiple_sources(self, link_type):
        link = link_type([times2_cu, times4_cu])
        times4 = cuda.declare_device("times4", "int32(int32)", link=link)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = times4(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 4)

    def test_link_multiple_sources_set(self):
        self._test_link_multiple_sources(set)

    def test_link_multiple_sources_tuple(self):
        self._test_link_multiple_sources(tuple)

    def test_link_multiple_sources_list(self):
        self._test_link_multiple_sources(list)

    @skip_unless_cffi
    def test_link_sources_in_memory_and_on_disk(self):
        jitlink_cu = str(test_data_dir / "jitlink.cu")
        link = [jitlink_cu, jitlink_user_cu]
        sig = types.void(types.CPointer(types.int32))
        ext_fn = cuda.declare_device("use_array_mutator", sig, link=link)

        ffi = cffi.FFI()

        @cuda.jit
        def kernel(x):
            ptr = ffi.from_buffer(x)
            ext_fn(ptr)

        x = np.arange(2, dtype=np.int32)
        kernel[1, 1](x)

        expected = np.ones(2, dtype=np.int32)
        np.testing.assert_equal(x, expected)

    @skip_if_curand_kernel_missing
    def test_include_cuda_header(self):
        sig = types.int32(types.uint64)
        link = [rng_cu]
        random_number = cuda.declare_device("random_number", sig, link=link)

        @cuda.jit
        def kernel(x, seed):
            x[0] = random_number(seed)

        x = np.zeros(1, dtype=np.uint32)
        kernel[1, 1](x, 1)
        np.testing.assert_equal(x[0], 323845807)

    def test_declared_in_called_function(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def device_func(x):
            return times2(x)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = device_func(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)

    def test_declared_in_called_function_twice(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def device_func_1(x):
            return times2(x)

        @cuda.jit
        def device_func_2(x):
            return device_func_1(x)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = device_func_2(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)

    def test_declared_in_called_function_two_calls(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def device_func(x):
            return times2(x)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = device_func(x[i]) + device_func(x[i] + i)

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 6)

    def test_call_declared_function_twice(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = times2(x[i]) + times2(x[i] + i)

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 6)

    def test_declared_in_called_function_and_parent(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)

        @cuda.jit
        def device_func(x):
            return times2(x)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = device_func(x[i]) + times2(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 4)

    def test_call_two_different_declared_functions(self):
        times2 = cuda.declare_device("times2", "int32(int32)", link=times2_cu)
        times3 = cuda.declare_device("times3", "int32(int32)", link=times3_cu)

        @cuda.jit
        def kernel(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = times2(x[i]) + times3(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        kernel[1, 32](r, x)

        np.testing.assert_equal(r, x * 5)


if __name__ == "__main__":
    unittest.main()
