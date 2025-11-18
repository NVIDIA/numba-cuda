# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.cudadrv.driver import _have_nvjitlink
from llvmlite import ir

import numpy as np
import os
from numba import cuda
from numba.cuda import HAS_NUMBA
from numba.cuda.testing import skip_on_standalone_numba_cuda, skip_on_cudasim
from numba.cuda import types
from numba.cuda import config

if config.ENABLE_CUDASIM:
    raise unittest.SkipTest("Simulator does not support extending types")

import inspect
import math
import pickle
import unittest

from numba.cuda.dispatcher import register_arg_handler

import numba
from numba import njit
from numba.cuda import cgutils, jit
from numba.cuda.tests.support import TestCase, override_config
from numba.cuda.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
from numba.cuda.typing.typeof import typeof

from numba.core import errors
from numba.cuda.errors import LoweringError

from numba.cuda.extending import (
    type_callable,
    lower_builtin,
    overload,
    overload_method,
    intrinsic,
    _Intrinsic,
    register_jitable,
    core_models,
    typeof_impl,
    register_model,
    make_attribute_wrapper,
)
from numba.cuda import dispatcher

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


class Interval:
    """
    A half-open interval on the real number line.
    """

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return "Interval(%f, %f)" % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


if HAS_NUMBA:
    from numba import njit
else:
    njit = None


@njit
def interval_width(interval):
    return interval.width


@njit
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)


class IntervalType(types.Type):
    def __init__(self):
        super().__init__(name="Interval")


interval_type = IntervalType()


@typeof_impl.register(Interval)
def typeof_interval(val, c):
    return interval_type


@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type

    return typer


@register_model(IntervalType)
class IntervalModel(core_models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("lo", types.float64),
            ("hi", types.float64),
        ]
        core_models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntervalType, "lo", "lo")
make_attribute_wrapper(IntervalType, "hi", "hi")


@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    return interval._getvalue()


@cuda_registry.register_attr
class Interval_attrs(AttributeTemplate):
    key = IntervalType

    def resolve_width(self, mod):
        return types.float64


@cuda_lower_attr(IntervalType, "width")
def cuda_Interval_width(context, builder, sig, arg):
    lo = builder.extract_value(arg, 0)
    hi = builder.extract_value(arg, 1)
    return builder.fsub(hi, lo)


# -----------------------------------------------------------------------
# Define a function's typing and implementation using the classical
# two-step API


def func1(x=None):
    raise NotImplementedError


def type_func1_(context):
    def typer(x=None):
        if x in (None, types.none):
            # 0-arg or 1-arg with None
            return types.int32
        elif isinstance(x, types.Float):
            # 1-arg with float
            return x

    return typer


type_func1 = type_callable(func1)(type_func1_)


@lower_builtin(func1)
@lower_builtin(func1, types.none)
def func1_nullary(context, builder, sig, args):
    return context.get_constant(sig.return_type, 42)


@lower_builtin(func1, types.Float)
def func1_unary(context, builder, sig, args):
    def func1_impl(x):
        return math.sqrt(2 * x)

    return context.compile_internal(builder, func1_impl, sig, args)


# -----------------------------------------------------------------------
# Overload an already defined built-in function, extending it for new types.


def call_func1_nullary(res):
    res[0] = func1()


def call_func1_unary(x, res):
    res[0] = func1(x)


class TestExtending(CUDATestCase):
    def test_attributes(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.lo
            r[1] = iv.hi

        x = np.asarray((1.5, 2.5))
        r = np.zeros_like(x)

        f[1, 1](r, x)

        np.testing.assert_equal(r, x)

    def test_property(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.width

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    @skip_on_standalone_numba_cuda
    def test_extension_type_as_arg(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = interval_width(iv)

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    @skip_on_standalone_numba_cuda
    def test_extension_type_as_retvalue(self):
        @cuda.jit
        def f(r, x):
            iv1 = Interval(x[0], x[1])
            iv2 = Interval(x[2], x[3])
            iv_sum = sum_intervals(iv1, iv2)
            r[0] = iv_sum.lo
            r[1] = iv_sum.hi

        x = np.asarray((1.5, 2.5, 3.0, 4.0))
        r = np.zeros(2)

        f[1, 1](r, x)

        expected = np.asarray((x[0] + x[2], x[1] + x[3]))
        np.testing.assert_allclose(r, expected)


class TestExtendingLinkage(CUDATestCase):
    @unittest.skipUnless(TEST_BIN_DIR, "Necessary binaries are not available")
    def test_extension_adds_linkable_code(self):
        files = (
            (test_device_functions_a, cuda.Archive),
            (test_device_functions_cubin, cuda.Cubin),
            (test_device_functions_cu, cuda.CUSource),
            (test_device_functions_fatbin, cuda.Fatbin),
            (test_device_functions_o, cuda.Object),
            (test_device_functions_ptx, cuda.PTXSource),
            (test_device_functions_ltoir, cuda.LTOIR),
        )

        lto = _have_nvjitlink()

        for path, ctor in files:
            if ctor == cuda.LTOIR and not lto:
                # Don't try to test with LTOIR if LTO is not enabled
                continue

            with open(path, "rb") as f:
                code_object = ctor(f.read())

            def external_add(x, y):
                return x + y

            @type_callable(external_add)
            def type_external_add(context):
                def typer(x, y):
                    if x == types.uint32 and y == types.uint32:
                        return types.uint32

                return typer

            @lower_builtin(external_add, types.uint32, types.uint32)
            def lower_external_add(context, builder, sig, args):
                context.active_code_library.add_linking_file(code_object)
                i32 = ir.IntType(32)
                fnty = ir.FunctionType(i32, [i32, i32])
                fn = cgutils.get_or_insert_function(
                    builder.module, fnty, "add_cabi"
                )
                return builder.call(fn, args)

            @cuda.jit(lto=lto)
            def use_external_add(r, x, y):
                r[0] = external_add(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

            @cuda.jit(lto=lto)
            def use_external_add_device(x, y):
                return external_add(x, y)

            @cuda.jit(lto=lto)
            def use_external_add_kernel(r, x, y):
                r[0] = use_external_add_device(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add_kernel[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

    def test_linked_called_through_overload(self):
        cu_code = cuda.CUSource("""
            extern "C" __device__
            int bar(int *out, int a)
            {
              *out = a * 2;
              return 0;
            }
        """)

        bar = cuda.declare_device("bar", "int32(int32)", link=cu_code)

        def bar_call(val):
            pass

        @overload(bar_call, target="cuda")
        def ol_bar_call(a):
            return lambda a: bar(a)

        @cuda.jit("void(int32[::1], int32[::1])")
        def foo(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = bar_call(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        foo[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)


@skip_on_cudasim("Extensions not supported in the simulator")
class TestArgHandlerRegistration(CUDATestCase):
    def test_register_arg_handler(self):
        class NumpyArrayWrapper_int32:
            def __init__(self, arr):
                self.arr = arr

        def numpy_array_wrapper_int32_arg_handler(ty, val, **kwargs):
            return types.int32[::1], val.arr

        def numpy_array_wrapper_int32_typeof_impl(val, c):
            return typeof(val.arr)

        register_arg_handler(
            numpy_array_wrapper_int32_arg_handler,
            (NumpyArrayWrapper_int32,),
            numpy_array_wrapper_int32_typeof_impl,
        )

        @cuda.jit("void(int32[::1])")
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] += 1

        arr = np.zeros(10, dtype=np.int32)
        wrapped_arr = NumpyArrayWrapper_int32(arr)

        kernel.forall(len(arr))(wrapped_arr)
        np.testing.assert_equal(arr, np.ones(10, dtype=np.int32))

        # No signature case
        @cuda.jit
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] += 1

        arr = np.zeros(10, dtype=np.int32)
        wrapped_arr = NumpyArrayWrapper_int32(arr)
        kernel.forall(len(arr))(wrapped_arr)

        np.testing.assert_equal(arr, np.ones(10, dtype=np.int32))

        # Two arg handlers
        class NumpyArrayWrapper_float32:
            def __init__(self, arr):
                self.arr = arr

        def numpy_array_wrapper_float32_arg_handler(ty, val, **kwargs):
            return types.float32[::1], val.arr

        def numpy_array_wrapper_float32_typeof_impl(val, c):
            return typeof(val.arr)

        register_arg_handler(
            numpy_array_wrapper_float32_arg_handler,
            (NumpyArrayWrapper_float32,),
            numpy_array_wrapper_float32_typeof_impl,
        )

        @cuda.jit("void(float32[::1], int32[::1])")
        def kernel(arr_f, arr_i):
            i = cuda.grid(1)
            if i < arr_f.size:
                arr_f[i] += 1.0
                arr_i[i] += 2

        arr_f = np.zeros(10, dtype=np.float32)
        arr_i = np.zeros(10, dtype=np.int32)
        wrapped_arr_f = NumpyArrayWrapper_float32(arr_f)
        wrapped_arr_i = NumpyArrayWrapper_int32(arr_i)
        kernel.forall(len(arr_f))(wrapped_arr_f, wrapped_arr_i)
        np.testing.assert_equal(arr_f, np.ones(10, dtype=np.float32))
        np.testing.assert_equal(arr_i, np.ones(10, dtype=np.int32) * 2)

        # multiple handlers for the same type - last one wins
        def numpy_array_wrapper_int32_arg_handler_v2(ty, val, **kwargs):
            return types.float64[::1], val.arr

        def numpy_array_wrapper_int32_typeof_impl_v2(val, c):
            return typeof(val.arr)

        register_arg_handler(
            numpy_array_wrapper_int32_arg_handler_v2,
            (NumpyArrayWrapper_int32,),
            numpy_array_wrapper_int32_typeof_impl_v2,
        )

        @cuda.jit("void(float64[::1])")
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] += 3.0

        arr = np.zeros(10, dtype=np.float64)
        wrapped_arr = NumpyArrayWrapper_int32(arr)
        kernel.forall(len(arr))(wrapped_arr)
        np.testing.assert_equal(arr, np.ones(10, dtype=np.float64) * 3.0)

        # Register one pass one
        # clear all handlers
        dispatcher._arg_handlers = {}
        register_arg_handler(
            numpy_array_wrapper_int32_arg_handler,
            (NumpyArrayWrapper_int32,),
            numpy_array_wrapper_int32_typeof_impl,
        )

        class PassedArgHandler_float32:
            def prepare_args(self, ty, val, **kwargs):
                return types.float32[::1], val.arr

        @cuda.jit(extensions=[PassedArgHandler_float32()])
        def kernel(arr_i, arr_f):
            i = cuda.grid(1)
            if i < arr_i.size:
                arr_i[i] += 4
                arr_f[i] += 5.0

        arr_i = np.zeros(10, dtype=np.int32)
        arr_f = np.zeros(10, dtype=np.float32)

        wrapped_arr_i = NumpyArrayWrapper_int32(arr_i)
        wrapped_arr_f = NumpyArrayWrapper_float32(arr_f)
        kernel.forall(len(arr_i))(wrapped_arr_i, wrapped_arr_f)
        np.testing.assert_equal(arr_i, np.ones(10, dtype=np.int32) * 4)
        np.testing.assert_equal(arr_f, np.ones(10, dtype=np.float32) * 5.0)


@skip_on_cudasim("Extensions not supported in the simulator")
class TestLowLevelExtending(TestCase):
    """
    Test the low-level two-tier extension API.
    """

    # Check with `@jit` from within the test process and also in a new test
    # process so as to check the registration mechanism.

    def test_func1(self):
        pyfunc = call_func1_nullary
        cfunc = jit(pyfunc)
        res = np.zeros(1)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            cfunc[1, 1](res)
        self.assertPreciseEqual(res[0], 42.0)
        pyfunc = call_func1_unary
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            cfunc = jit(pyfunc)
        self.assertPreciseEqual(res[0], 42.0)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            cfunc[1, 1](18.0, res)
        self.assertPreciseEqual(res[0], 6.0)

    @TestCase.run_test_in_subprocess
    def test_func1_isolated(self):
        self.test_func1()

    def test_type_callable_keeps_function(self):
        self.assertIs(type_func1, type_func1_)
        self.assertIsNotNone(type_func1)


class TestHighLevelExtending(TestCase):
    """
    Test the high-level combined API.
    """

    def test_typing_vs_impl_signature_mismatch_handling(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception.
        """

        def gen_ol(impl=None):
            def myoverload(a, b, c, kw=None):
                pass

            @overload(myoverload)
            def _myoverload_impl(a, b, c, kw=None):
                return impl

            @jit
            def foo(a, b, c, d):
                myoverload(a, b, c, kw=d)

            return foo

        sentinel = "Typing and implementation arguments differ in"

        # kwarg value is different
        def impl1(a, b, c, kw=12):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl1)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument default values", msg)
        self.assertIn('<Parameter "kw=12">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        # kwarg name is different
        def impl2(a, b, c, kwarg=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl2)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "kwarg=None">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        # arg name is different
        def impl3(z, b, c, kw=None):
            if a > 10:  # noqa: F821
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl3)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)

        from .overload_usecases import impl4, impl5

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl4)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn("First difference: 'z'", msg)

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl5)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)

        # too many args
        def impl6(a, b, c, d, e, kw=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl6)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "d">', msg)
        self.assertIn('<Parameter "e">', msg)

        # too few args
        def impl7(a, b, kw=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl7)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "c">', msg)

        # too many kwargs
        def impl8(a, b, c, kw=None, extra_kwarg=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl8)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "extra_kwarg=None">', msg)

        # too few kwargs
        def impl9(a, b, c):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(impl9)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "kw=None">', msg)

    def test_typing_vs_impl_signature_mismatch_handling_var_positional(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception and uses VAR_POSITIONAL (*args) in typing
        """

        def myoverload(a, kw=None):
            pass

        from .overload_usecases import var_positional_impl

        overload(myoverload)(var_positional_impl)

        @jit
        def foo(a, b):
            myoverload(a, b, 9, kw=11)

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                foo[1, 1](1, 5)
        msg = str(e.exception)
        self.assertIn("VAR_POSITIONAL (e.g. *args) argument kind", msg)
        self.assertIn("offending argument name is '*star_args_token'", msg)

    def test_typing_vs_impl_signature_mismatch_handling_var_keyword(self):
        """
        Tests that an overload which uses **kwargs (VAR_KEYWORD)
        """

        def gen_ol(impl, strict=True):
            def myoverload(a, kw=None):
                pass

            overload(myoverload, strict=strict)(impl)

            @jit
            def foo(a, b):
                myoverload(a, kw=11)

            return foo

        # **kwargs in typing
        def ol1(a, **kws):
            def impl(a, kw=10):
                return a

            return impl

        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            gen_ol(ol1, False)[1, 1](
                1, 2
            )  # no error if strictness not enforced
        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(ol1)[1, 1](1, 2)
        msg = str(e.exception)
        self.assertIn("use of VAR_KEYWORD (e.g. **kwargs) is unsupported", msg)
        self.assertIn("offending argument name is '**kws'", msg)

        # **kwargs in implementation
        def ol2(a, kw=0):
            def impl(a, **kws):
                return a

            return impl

        with self.assertRaises(errors.TypingError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                gen_ol(ol2)[1, 1](1, 2)
        msg = str(e.exception)
        self.assertIn("use of VAR_KEYWORD (e.g. **kwargs) is unsupported", msg)
        self.assertIn("offending argument name is '**kws'", msg)

    def test_overload_method_kwargs(self):
        # Issue #3489
        @overload_method(types.Array, "foo")
        def fooimpl(arr, a_kwarg=10):
            def impl(arr, a_kwarg=10):
                return a_kwarg

            return impl

        @jit
        def bar(A, res):
            res[0] = A.foo()
            res[1] = A.foo(20)
            res[2] = A.foo(a_kwarg=30)

        Z = np.arange(5)
        res = np.zeros(3)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            bar[1, 1](Z, res)
        self.assertEqual(res[0], 10)
        self.assertEqual(res[1], 20)
        self.assertEqual(res[2], 30)

    def test_overload_method_literal_unpack(self):
        # Issue #3683
        @overload_method(types.Array, "litfoo")
        def litfoo(arr, val):
            # Must be an integer
            if isinstance(val, types.Integer):
                # Must not be literal
                if not isinstance(val, types.Literal):

                    def impl(arr, val):
                        return val

                    return impl

        @jit
        def bar(A, res):
            res[0] = A.litfoo(0xCAFE)

        A = np.zeros(1)
        res = np.zeros(1)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            bar[1, 1](A, res)
        self.assertEqual(res[0], 0xCAFE)


def _assert_cache_stats(cfunc, expect_hit, expect_misses):
    hit = cfunc._cache_hits[cfunc.signatures[0]]
    if hit != expect_hit:
        raise AssertionError("cache not used")
    miss = cfunc._cache_misses[cfunc.signatures[0]]
    if miss != expect_misses:
        raise AssertionError("cache not used")


class TestIntrinsic(TestCase):
    def test_void_return(self):
        """
        Verify that returning a None from codegen function is handled
        automatically for void functions, otherwise raise exception.
        """

        @intrinsic
        def void_func(typingctx, a):
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass  # do nothing, return None, should be turned into
                # dummy value

            return sig, codegen

        @intrinsic
        def non_void_func(typingctx, a):
            sig = types.int32(types.int32)

            def codegen(context, builder, signature, args):
                pass  # oops, should be returning a value here, raise exception

            return sig, codegen

        @jit
        def call_void_func():
            void_func(1)

        @jit
        def call_non_void_func():
            non_void_func(1)

        # void func should work
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            self.assertEqual(call_void_func[1, 1](), None)
        # not void function should raise exception
        with self.assertRaises(LoweringError) as e:
            with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
                call_non_void_func[1, 1]()
        self.assertIn("non-void function returns None", e.exception.msg)

    def test_serialization(self):
        """
        Test serialization of intrinsic objects
        """

        # define a intrinsic
        @intrinsic
        def identity(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            sig = x(x)
            return sig, codegen

        # use in a jit function
        @jit
        def foo(x):
            identity(x)

        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            self.assertEqual(foo[1, 1](1), None)

        # get serialization memo
        memo = _Intrinsic._memo
        memo_size = len(memo)

        # pickle foo and check memo size
        serialized_foo = pickle.dumps(foo)
        # increases the memo size
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        # unpickle
        foo_rebuilt = pickle.loads(serialized_foo)
        self.assertEqual(memo_size, len(memo))
        # check rebuilt foo

        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            self.assertEqual(foo[1, 1](1), foo_rebuilt[1, 1](1))

        # pickle identity directly
        serialized_identity = pickle.dumps(identity)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))
        # unpickle
        identity_rebuilt = pickle.loads(serialized_identity)
        # must be the same object
        self.assertIs(identity, identity_rebuilt)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))

    def test_deserialization(self):
        """
        Test deserialization of intrinsic
        """

        def defn(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            return x(x), codegen

        memo = _Intrinsic._memo
        memo_size = len(memo)
        # invoke _Intrinsic indirectly to avoid registration which keeps an
        # internal reference inside the compiler
        original = _Intrinsic("foo", defn)
        self.assertIs(original._defn, defn)
        pickled = pickle.dumps(original)
        # by pickling, a new memo entry is created
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        del original  # remove original before unpickling

        # by deleting, the memo entry is NOT removed due to recent
        # function queue
        self.assertEqual(memo_size, len(memo))

        # Manually force clear of _recent queue
        _Intrinsic._recent.clear()
        memo_size -= 1
        self.assertEqual(memo_size, len(memo))

        rebuilt = pickle.loads(pickled)
        # verify that the rebuilt object is different
        self.assertIsNot(rebuilt._defn, defn)

        # the second rebuilt object is the same as the first
        second = pickle.loads(pickled)
        self.assertIs(rebuilt._defn, second._defn)

    def test_docstring(self):
        @intrinsic
        def void_func(typingctx, a: int):
            """void_func docstring"""
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass  # do nothing, return None, should be turned into
                # dummy value

            return sig, codegen

        self.assertEqual(
            "numba.cuda.tests.cudapy.test_extending", void_func.__module__
        )
        self.assertEqual("void_func", void_func.__name__)
        self.assertEqual(
            "TestIntrinsic.test_docstring.<locals>.void_func",
            void_func.__qualname__,
        )
        self.assertDictEqual({"a": int}, void_func.__annotations__)
        self.assertEqual("void_func docstring", void_func.__doc__)


class TestRegisterJitable(unittest.TestCase):
    def test_no_flags(self):
        @register_jitable
        def foo(x, y):
            x[0] += y

        def bar(x, y):
            foo(x, y)
            x[0] += x[0]

        cbar = jit(bar)

        x = np.array([1, 2])
        bar(x, 2)
        self.assertEqual(x[0], 6)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            cbar[1, 1](x, 2)
        self.assertEqual(x[0], 16)


class TestOverloadPreferLiteral(TestCase):
    def test_overload(self):
        def prefer_lit(x):
            pass

        def non_lit(x):
            pass

        def ov(x):
            if isinstance(x, types.IntegerLiteral):
                # With prefer_literal=False, this branch will not be reached.
                if x.literal_value == 1:

                    def impl(x):
                        return 0xCAFE

                    return impl
                else:
                    raise errors.TypingError("literal value")
            else:

                def impl(x):
                    return x * 100

                return impl

        overload(prefer_lit, prefer_literal=True)(ov)
        overload(non_lit)(ov)

        @jit
        def check_prefer_lit(x, res):
            res[0] = prefer_lit(1)
            res[1] = prefer_lit(2)
            res[2] = prefer_lit(x)

        res = np.zeros(3)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            check_prefer_lit[1, 1](3, res)
        a, b, c = res
        self.assertEqual(a, 0xCAFE)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)

        @jit
        def check_non_lit(x, res):
            res[0] = non_lit(1)
            res[1] = non_lit(2)
            res[2] = non_lit(x)

        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            check_non_lit[1, 1](3, res)
        a, b, c = res
        self.assertEqual(a, 100)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)


class TestNumbaInternalOverloads(TestCase):
    def test_signatures_match_overloaded_api(self):
        # This is a "best-effort" test to try and ensure that Numba's internal
        # overload declarations have signatures with argument names that match
        # the API they are overloading. The purpose of ensuring there is a
        # match is so that users can use call-by-name for positional arguments.

        # Set this to:
        # 0 to make violations raise a ValueError (default).
        # 1 to get violations reported to STDOUT
        # 2 to get a verbose output of everything that was checked and its state
        #   reported to STDOUT.
        DEBUG = 0

        # np.random.* does not have a signature exposed to `inspect`... so
        # custom parse the docstrings.
        def sig_from_np_random(x):
            if not x.startswith("_"):
                thing = getattr(np.random, x)
                if inspect.isbuiltin(thing):
                    docstr = thing.__doc__.splitlines()
                    for l in docstr:
                        if l:
                            sl = l.strip()
                            if sl.startswith(x):  # its the signature
                                # special case np.random.seed, it has `self` in
                                # the signature whereas all the other functions
                                # do not!?
                                if x == "seed":
                                    sl = "seed(seed)"

                                fake_impl = f"def {sl}:\n\tpass"
                                l = {}
                                try:
                                    exec(fake_impl, {}, l)
                                except SyntaxError:
                                    # likely elipsis, e.g. rand(d0, d1, ..., dn)
                                    if DEBUG == 2:
                                        print(
                                            "... skipped as cannot parse "
                                            "signature"
                                        )
                                    return None
                                else:
                                    fn = l.get(x)
                                    return inspect.signature(fn)

        def checker(func, overload_func):
            if DEBUG == 2:
                print(f"Checking: {func}")

            def create_message(func, overload_func, func_sig, ol_sig):
                msg = []
                s = (
                    f"{func} from module '{getattr(func, '__module__')}' "
                    "has mismatched sig."
                )
                msg.append(s)
                msg.append(f"    - expected: {func_sig}")
                msg.append(f"    -      got: {ol_sig}")
                lineno = inspect.getsourcelines(overload_func)[1]
                tmpsrcfile = inspect.getfile(overload_func)
                srcfile = tmpsrcfile.replace(numba.__path__[0], "")
                msg.append(f"from {srcfile}:{lineno}")
                msgstr = "\n" + "\n".join(msg)
                return msgstr

            func_sig = None
            try:
                func_sig = inspect.signature(func)
            except ValueError:
                # probably a built-in/C code, see if it's a np.random function
                if fname := getattr(func, "__name__", False):
                    if maybe_func := getattr(np.random, fname, False):
                        if maybe_func == func:
                            # it's a built-in from np.random
                            func_sig = sig_from_np_random(fname)

            if func_sig is not None:
                ol_sig = inspect.signature(overload_func)
                x = list(func_sig.parameters.keys())
                y = list(ol_sig.parameters.keys())
                for a, b in zip(x[: len(y)], y):
                    if a != b:
                        p = func_sig.parameters[a]
                        if p.kind == p.POSITIONAL_ONLY:
                            # probably a built-in/C code
                            if DEBUG == 2:
                                print(
                                    "... skipped as positional only "
                                    "arguments found"
                                )
                            break
                        elif "*" in str(p):  # probably *args or similar
                            if DEBUG == 2:
                                print("... skipped as contains *args")
                            break
                        else:
                            # Only error/report on functions that have a module
                            # or are from somewhere other than Numba.
                            if (
                                not func.__module__
                                or not func.__module__.startswith("numba")
                            ):
                                msgstr = create_message(
                                    func, overload_func, func_sig, ol_sig
                                )
                                if DEBUG != 0:
                                    if DEBUG == 2:
                                        print("... INVALID")
                                    if msgstr:
                                        print(msgstr)
                                    break
                                else:
                                    raise ValueError(msgstr)
                            else:
                                if DEBUG == 2:
                                    if not func.__module__:
                                        print(
                                            "... skipped as no __module__ "
                                            "present"
                                        )
                                    else:
                                        print("... skipped as Numba internal")
                                break
                else:
                    if DEBUG == 2:
                        print("... OK")

        # Compile something to make sure that the typing context registries
        # are populated with everything from the CPU target.
        jit(lambda: None).compile(())
        tyctx = numba.cuda.target.CUDATypingContext()
        tyctx.refresh()

        # Walk the registries and check each function that is an overload
        regs = tyctx._registries
        for k, v in regs.items():
            for item in k.functions:
                if getattr(item, "_overload_func", False):
                    checker(item.key, item._overload_func)


if __name__ == "__main__":
    unittest.main()
