# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda
from numba.cuda import types
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba.core.errors import TypingError
    from numba import njit
    import numba
else:
    from numba.cuda.core.errors import TypingError
from numba.cuda.extending import overload, overload_attribute
from numba.cuda.typing.typeof import typeof
from numba.core.typing.typeof import typeof as cpu_typeof
from numba.cuda.testing import (
    CUDATestCase,
    skip_on_cudasim,
    unittest,
    skip_on_standalone_numba_cuda,
)
import numpy as np


# Dummy function definitions to overload


def generic_func_1():
    pass


def cuda_func_1():
    pass


def generic_func_2():
    pass


def cuda_func_2():
    pass


def generic_calls_generic():
    pass


def generic_calls_cuda():
    pass


def cuda_calls_generic():
    pass


def cuda_calls_cuda():
    pass


def target_overloaded():
    pass


def generic_calls_target_overloaded():
    pass


def cuda_calls_target_overloaded():
    pass


def target_overloaded_calls_target_overloaded():
    pass


def default_values_and_kwargs():
    pass


# To recognise which functions are resolved for a call, we identify each with a
# prime number. Each function called multiplies a value by its prime (starting
# with the value 1), and we can check that the result is as expected based on
# the final value after all multiplications.

GENERIC_FUNCTION_1 = 2
CUDA_FUNCTION_1 = 3
GENERIC_FUNCTION_2 = 5
CUDA_FUNCTION_2 = 7
GENERIC_CALLS_GENERIC = 11
GENERIC_CALLS_CUDA = 13
CUDA_CALLS_GENERIC = 17
CUDA_CALLS_CUDA = 19
GENERIC_TARGET_OL = 23
CUDA_TARGET_OL = 29
GENERIC_CALLS_TARGET_OL = 31
CUDA_CALLS_TARGET_OL = 37
GENERIC_TARGET_OL_CALLS_TARGET_OL = 41
CUDA_TARGET_OL_CALLS_TARGET_OL = 43


# Overload implementations


@overload(generic_func_1, target="generic")
def ol_generic_func_1(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_1

    return impl


@overload(cuda_func_1, target="cuda")
def ol_cuda_func_1(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_1

    return impl


@overload(generic_func_2, target="generic")
def ol_generic_func_2(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_2

    return impl


@overload(cuda_func_2, target="cuda")
def ol_cuda_func(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_2

    return impl


@overload(generic_calls_generic, target="generic")
def ol_generic_calls_generic(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_GENERIC
        generic_func_1(x)

    return impl


@overload(generic_calls_cuda, target="generic")
def ol_generic_calls_cuda(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_CUDA
        cuda_func_1(x)

    return impl


@overload(cuda_calls_generic, target="cuda")
def ol_cuda_calls_generic(x):
    def impl(x):
        x[0] *= CUDA_CALLS_GENERIC
        generic_func_1(x)

    return impl


@overload(cuda_calls_cuda, target="cuda")
def ol_cuda_calls_cuda(x):
    def impl(x):
        x[0] *= CUDA_CALLS_CUDA
        cuda_func_1(x)

    return impl


@overload(target_overloaded, target="generic")
def ol_target_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_TARGET_OL

    return impl


@overload(target_overloaded, target="cuda")
def ol_target_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_TARGET_OL

    return impl


@overload(generic_calls_target_overloaded, target="generic")
def ol_generic_calls_target_overloaded(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(cuda_calls_target_overloaded, target="cuda")
def ol_cuda_calls_target_overloaded(x):
    def impl(x):
        x[0] *= CUDA_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(target_overloaded_calls_target_overloaded, target="generic")
def ol_generic_calls_target_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(target_overloaded_calls_target_overloaded, target="cuda")
def ol_generic_calls_target_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(default_values_and_kwargs)
def ol_default_values_and_kwargs(out, x, y=5, z=6):
    def impl(out, x, y=5, z=6):
        out[0], out[1] = x + y, z

    return impl


@skip_on_cudasim("Overloading not supported in cudasim")
class TestOverload(CUDATestCase):
    def check_overload(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        cuda.jit(kernel)[1, 1](x)
        self.assertEqual(x[0], expected)

    @skip_on_standalone_numba_cuda
    def check_overload_cpu(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        njit(kernel)(x)
        self.assertEqual(x[0], expected)

    def test_generic(self):
        def kernel(x):
            generic_func_1(x)

        expected = GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda(self):
        def kernel(x):
            cuda_func_1(x)

        expected = CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_and_cuda(self):
        def kernel(x):
            generic_func_1(x)
            cuda_func_1(x)

        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_two_generic_calls(self):
        def kernel(x):
            generic_func_1(x)
            generic_func_2(x)

        expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_call_two_cuda_calls(self):
        def kernel(x):
            cuda_func_1(x)
            cuda_func_2(x)

        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_generic_calls_generic(self):
        def kernel(x):
            generic_calls_generic(x)

        expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_calls_cuda(self):
        def kernel(x):
            generic_calls_cuda(x)

        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_generic(self):
        def kernel(x):
            cuda_calls_generic(x)

        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_cuda(self):
        def kernel(x):
            cuda_calls_cuda(x)

        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_target_overloaded(self):
        def kernel(x):
            target_overloaded(x)

        expected = CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_generic_calls_target_overloaded(self):
        def kernel(x):
            generic_calls_target_overloaded(x)

        expected = GENERIC_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_cuda_calls_target_overloaded(self):
        def kernel(x):
            cuda_calls_target_overloaded(x)

        expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_target_overloaded_calls_target_overloaded(self):
        def kernel(x):
            target_overloaded_calls_target_overloaded(x)

        # Check the CUDA overloads are used on CUDA
        expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    @skip_on_standalone_numba_cuda
    def test_target_overloaded_calls_target_overloaded_cpu(self):
        def kernel(x):
            target_overloaded_calls_target_overloaded(x)

        # Check that the CPU overloads are used on the CPU
        expected = GENERIC_TARGET_OL_CALLS_TARGET_OL * GENERIC_TARGET_OL
        self.check_overload_cpu(kernel, expected)

    @skip_on_standalone_numba_cuda
    def test_overload_attribute_target(self):
        MyDummy, MyDummyType = self.make_dummy_type()
        mydummy_type_cpu = cpu_typeof(MyDummy())  # For @njit (cpu)
        mydummy_type = typeof(MyDummy())  # For @cuda.jit (CUDA)

        @overload_attribute(MyDummyType, "cuda_only", target="cuda")
        def ov_dummy_cuda_attr(obj):
            def imp(obj):
                return 42

            return imp

        # Ensure that we cannot use the CUDA target-specific attribute on the
        # CPU, and that an appropriate typing error is raised

        # A different error is produced prior to version 0.60
        # (the fixes in #9454 improved the message)
        # https://github.com/numba/numba/pull/9454
        if HAS_NUMBA and numba.version_info[:2] < (0, 60):
            msg = 'resolving type of attribute "cuda_only" of "x"'
        else:
            msg = "Unknown attribute 'cuda_only'"

        with self.assertRaisesRegex(TypingError, msg):

            @njit(types.int64(mydummy_type_cpu))
            def illegal_target_attr_use(x):
                return x.cuda_only

        # Ensure that the CUDA target-specific attribute is usable and works
        # correctly when the target is CUDA - note eager compilation via
        # signature
        @cuda.jit(types.void(types.int64[::1], mydummy_type))
        def cuda_target_attr_use(res, dummy):
            res[0] = dummy.cuda_only

    def test_default_values_and_kwargs(self):
        """
        Test default values and kwargs.
        """

        @cuda.jit()
        def kernel(a, b, out):
            default_values_and_kwargs(out, a, z=b)

        out = np.empty(2, dtype=np.int64)
        kernel[1, 1](1, 2, out)
        self.assertEqual(tuple(out), (6, 2))


@skip_on_cudasim("Overloading not supported in cudasim")
class TestOverloadFuncCaching(CUDATestCase):
    """The overload body must execute at most once per argument-type set.

    Numba resolves the same overloaded call under several ConfigStack flag
    contexts during a single kernel compilation (type inference vs. a
    force-inlined / LTO device-function compile).  ``_impl_cache`` keys on those
    flags, so the -- potentially very expensive -- overload body would otherwise
    run once per context.  ``_OverloadFunctionTemplate`` memoizes the overload
    result (which depends only on the argument types, never on the flags) to
    collapse those to a single execution.  These tests pin that down.
    """

    @staticmethod
    def _make_template(overload_func, inline="never"):
        from numba.cuda.typing.templates import make_overload_template

        def target(x):
            pass

        return make_overload_template(
            target, overload_func, jit_options={}, strict=True, inline=inline
        )

    def test_overload_body_runs_once(self):
        calls = []

        def ol(x):
            calls.append(x)

            def impl(x):
                pass

            return impl

        template = self._make_template(ol)(None)
        argty = types.int32

        r1 = template._call_overload_func((argty,), {})
        r2 = template._call_overload_func((argty,), {})

        self.assertEqual(len(calls), 1)
        self.assertIs(r1, r2)

    def test_distinct_arg_types_run_again(self):
        calls = []

        def ol(x):
            calls.append(x)

            def impl(x):
                pass

            return impl

        template = self._make_template(ol)(None)

        template._call_overload_func((types.int32,), {})
        template._call_overload_func((types.int64,), {})

        self.assertEqual(len(calls), 2)

    def test_kwargs_participate_in_key(self):
        calls = []

        def ol(x, flag=False):
            calls.append((x, flag))

            def impl(x, flag=False):
                pass

            return impl

        template = self._make_template(ol)(None)

        template._call_overload_func((types.int32,), {})
        template._call_overload_func((types.int32,), {})
        template._call_overload_func((types.int32,), {"flag": True})

        self.assertEqual(len(calls), 2)

    def test_cache_is_per_template(self):
        calls_a = []
        calls_b = []

        def ol_a(x):
            calls_a.append(x)

            def impl(x):
                pass

            return impl

        def ol_b(x):
            calls_b.append(x)

            def impl(x):
                pass

            return impl

        template_a = self._make_template(ol_a)(None)
        template_b = self._make_template(ol_b)(None)
        argty = types.int32

        ra1 = template_a._call_overload_func((argty,), {})
        ra2 = template_a._call_overload_func((argty,), {})
        rb1 = template_b._call_overload_func((argty,), {})

        # Same argument type, but each template keeps its own cache: the two
        # distinct overloads must not alias one another.
        self.assertEqual(len(calls_a), 1)
        self.assertEqual(len(calls_b), 1)
        self.assertIs(ra1, ra2)
        self.assertIsNot(ra1, rb1)

    def test_cache_lives_on_template_class(self):
        # Template instances are transient -- Numba creates a fresh one per
        # resolution -- so the cache must live on the template *class* for a
        # second instance to reuse the first's result.
        calls = []

        def ol(x):
            calls.append(x)

            def impl(x):
                pass

            return impl

        template_cls = self._make_template(ol)
        argty = types.int32

        template_cls(None)._call_overload_func((argty,), {})
        template_cls(None)._call_overload_func((argty,), {})

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
