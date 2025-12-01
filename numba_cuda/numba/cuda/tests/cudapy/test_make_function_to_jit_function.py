# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda
from numba.cuda.core import errors
from numba.cuda.extending import overload
from numba.cuda.testing import skip_on_cudasim
import numpy as np

import unittest


@cuda.jit
def consumer(func, *args):
    return func(*args)


@cuda.jit
def consumer2arg(func1, func2):
    return func2(func1)


def wrap_with_kernel_noarg(func):
    jitted_func = cuda.jit(func)

    @cuda.jit
    def kernel(out):
        out[0] = jitted_func()

    def runner():
        out = np.zeros(1, dtype=np.int64)
        kernel[1, 1](out)
        return out[0]

    return runner


def wrap_with_kernel_one_arg(func):
    jitted_func = cuda.jit(func)

    @cuda.jit
    def kernel(out, in1):
        out[0] = jitted_func(in1)

    def runner(in1):
        out = np.zeros(1, dtype=np.int64)
        kernel[1, 1](out, in1)
        return out[0]

    return runner


def wrap_with_kernel_two_args(func):
    jitted_func = cuda.jit(func)

    @cuda.jit
    def kernel(out, in1, in2):
        out[0] = jitted_func(in1, in2)

    def runner(in1, in2):
        out = np.zeros(1, dtype=np.int64)
        kernel[1, 1](out, in1, in2)
        return out[0]

    return runner


def wrap_with_kernel_noarg_tuple_return(func):
    jitted_func = cuda.jit(func)

    @cuda.jit
    def kernel(out):
        out[0], out[1], out[2], out[3] = jitted_func()

    def runner():
        out = np.zeros(4, dtype=np.int64)
        kernel[1, 1](out)
        return out[0], out[1], out[2], out[3]

    return runner


_global = 123


class TestMakeFunctionToJITFunction(unittest.TestCase):
    """
    This tests the pass that converts ir.Expr.op == make_function (i.e. closure)
    into a JIT function.
    """

    # NOTE: testing this is a bit tricky. The function receiving a JIT'd closure
    # must also be under JIT control so as to handle the JIT'd closure
    # correctly, however, in the case of running the test implementations in the
    # interpreter, the receiving function cannot be JIT'd else it will receive
    # the Python closure and then complain about pyobjects as arguments.
    # The way around this is to use a factory function to close over either the
    # jitted or standard python function as the consumer depending on context.

    def test_escape(self):
        def impl_factory(consumer_func):
            def impl():
                def inner():
                    return 10

                return consumer_func(inner)

            return impl

        cfunc = wrap_with_kernel_noarg(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_nested_escape(self):
        def impl_factory(consumer_func):
            def impl():
                def inner():
                    return 10

                def innerinner(x):
                    return x()

                return consumer_func(inner, innerinner)

            return impl

        cfunc = wrap_with_kernel_noarg(impl_factory(consumer2arg))
        impl = impl_factory(consumer2arg.py_func)

        self.assertEqual(impl(), cfunc())

    def test_closure_in_escaper(self):
        def impl_factory(consumer_func):
            def impl():
                def callinner():
                    def inner():
                        return 10

                    return inner()

                return consumer_func(callinner)

            return impl

        cfunc = wrap_with_kernel_noarg(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_close_over_consts(self):
        def impl_factory(consumer_func):
            def impl():
                y = 10

                def callinner(z):
                    return y + z + _global

                return consumer_func(callinner, 6)

            return impl

        cfunc = wrap_with_kernel_noarg(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_close_over_consts_w_args(self):
        def impl_factory(consumer_func):
            def impl(x):
                y = 10

                def callinner(z):
                    return y + z + _global

                return consumer_func(callinner, x)

            return impl

        cfunc = wrap_with_kernel_one_arg(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_with_overload(self):
        def foo(func, *args):
            nargs = len(args)
            if nargs == 1:
                return func(*args)
            elif nargs == 2:
                return func(func(*args))

        @overload(foo)
        def foo_ol(func, *args):
            # specialise on the number of args, as per `foo`
            nargs = len(args)
            if nargs == 1:

                def impl(func, *args):
                    return func(*args)

                return impl
            elif nargs == 2:

                def impl(func, *args):
                    return func(func(*args))

                return impl

        def impl_factory(consumer_func):
            def impl(x):
                y = 10

                def callinner(*z):
                    if len(z) == 1:
                        tmp = z[0]
                    elif len(z) == 2:
                        tmp = z[0] + z[1]
                    return y + tmp + _global

                # run both specialisations, 1 arg, and 2 arg.
                return foo(callinner, x) + foo(callinner, x, x)

            return impl

        cfunc = wrap_with_kernel_one_arg(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_basic_apply_like_case(self):
        def apply(arg, func):
            return func(arg)

        @overload(apply)
        def ov_apply(arg, func):
            return lambda arg, func: func(arg)

        def impl(arg):
            def mul10(x):
                return x * 10

            return apply(arg, mul10)

        cfunc = wrap_with_kernel_one_arg(impl)

        a = 10
        np.testing.assert_allclose(impl(a), cfunc(a))

    # this needs true SSA to be able to work correctly, check error for now
    @skip_on_cudasim("Simulator will not raise a typing error")
    def test_multiply_defined_freevar(self):
        def impl(c):
            if c:
                x = 3

                def inner(y):
                    return y + x

                r = consumer(inner, 1)
            else:
                x = 6

                def inner(y):
                    return y + x

                r = consumer(inner, 2)
            return r

        with self.assertRaises(errors.TypingError) as e:
            cuda.jit("void(int64)")(impl)

        self.assertIn(
            "Cannot capture a constant value for variable", str(e.exception)
        )

    @skip_on_cudasim("Simulator will not raise a typing error")
    def test_non_const_in_escapee(self):
        def impl(x):
            z = np.arange(x)

            def inner(val):
                return 1 + z + val  # z is non-const freevar

            return consumer(inner, x)

        with self.assertRaises(errors.TypingError) as e:
            cuda.jit("void(int64)")(impl)

        self.assertIn(
            "Cannot capture the non-constant value associated", str(e.exception)
        )

    def test_escape_with_kwargs(self):
        def impl_factory(consumer_func):
            def impl():
                t = 12

                def inner(a, b, c, mydefault1=123, mydefault2=456):
                    z = 4
                    return mydefault1 + mydefault2 + z + t + a + b + c

                # this is awkward, top and tail closure inlining with a escapees
                # in the middle that do/don't have defaults.
                return (
                    inner(1, 2, 5, 91, 53),
                    consumer_func(inner, 1, 2, 3, 73),
                    consumer_func(
                        inner,
                        1,
                        2,
                        3,
                    ),
                    inner(1, 2, 4),
                )

            return impl

        cfunc = wrap_with_kernel_noarg_tuple_return(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        np.testing.assert_allclose(impl(), cfunc())

    def test_escape_with_kwargs_override_kwargs(self):
        @cuda.jit
        def specialised_consumer(func, *args):
            x, y, z = args  # unpack to avoid `CALL_FUNCTION_EX`
            a = func(x, y, z, mydefault1=1000)
            b = func(x, y, z, mydefault2=1000)
            c = func(x, y, z, mydefault1=1000, mydefault2=1000)
            return a + b + c

        def impl_factory(consumer_func):
            def impl():
                t = 12

                def inner(a, b, c, mydefault1=123, mydefault2=456):
                    z = 4
                    return mydefault1 + mydefault2 + z + t + a + b + c

                # this is awkward, top and tail closure inlining with a escapees
                # in the middle that get defaults specified in the consumer
                return (
                    inner(1, 2, 5, 91, 53),
                    consumer_func(inner, 1, 2, 11),
                    consumer_func(
                        inner,
                        1,
                        2,
                        3,
                    ),
                    inner(1, 2, 4),
                )

            return impl

        cfunc = wrap_with_kernel_noarg_tuple_return(
            impl_factory(specialised_consumer)
        )
        impl = impl_factory(specialised_consumer.py_func)

        np.testing.assert_allclose(impl(), cfunc())
