# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""
Tests for SSA reconstruction
"""

import sys
import copy
import logging

import numpy as np

from numba import types, cuda
from numba.cuda import jit
from numba.core import errors

from numba.extending import overload
from numba.tests.support import override_config
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


_DEBUG = False

if _DEBUG:
    # Enable debug logger on SSA reconstruction
    ssa_logger = logging.getLogger("numba.core.ssa")
    ssa_logger.setLevel(level=logging.DEBUG)
    ssa_logger.addHandler(logging.StreamHandler(sys.stderr))


class SSABaseTest(CUDATestCase):
    """
    This class comes from numba tests, but has been modified to work with CUDA kernels.
    Return values were replaced by output arrays, and tuple returns assign to elements of the output array.
    """

    def check_func(self, func, result_array, *args):
        # For CUDA kernels, we need to create output arrays and call with [1,1] launch config
        # Create GPU array with same shape as expected result array
        gpu_result_array = cuda.to_device(np.zeros_like(result_array))

        # Call the CUDA kernel
        func[1, 1](gpu_result_array, *copy.deepcopy(args))
        gpu_result = gpu_result_array.copy_to_host()

        # Call the original Python function for expected result
        cpu_result = np.zeros_like(result_array)
        func.py_func(cpu_result, *copy.deepcopy(args))

        # Compare all results
        np.testing.assert_array_equal(gpu_result, cpu_result)


class TestSSA(SSABaseTest):
    """
    Contains tests to help isolate problems in SSA
    """

    def test_argument_name_reused(self):
        @jit
        def foo(result, x):
            x += 1
            result[0] = x

        self.check_func(foo, np.array([124.0]), 123)

    def test_if_else_redefine(self):
        @jit
        def foo(result, x, y):
            z = x * y
            if x < y:
                z = x
            else:
                z = y
            result[0] = z

        self.check_func(foo, np.array([2.0]), 3, 2)
        self.check_func(foo, np.array([2.0]), 2, 3)

    def test_sum_loop(self):
        @jit
        def foo(result, n):
            c = 0
            for i in range(n):
                c += i
            result[0] = c

        self.check_func(foo, np.array([0.0]), 0)
        self.check_func(foo, np.array([45.0]), 10)

    def test_sum_loop_2vars(self):
        @jit
        def foo(result, n):
            c = 0
            d = n
            for i in range(n):
                c += i
                d += n
            result[0] = c
            result[1] = d

        self.check_func(foo, np.array([0.0, 0.0]), 0)
        self.check_func(foo, np.array([45.0, 110.0]), 10)

    def test_sum_2d_loop(self):
        @jit
        def foo(result, n):
            c = 0
            for i in range(n):
                for j in range(n):
                    c += j
                c += i
            result[0] = c

        self.check_func(foo, np.array([0.0]), 0)
        self.check_func(foo, np.array([495.0]), 10)

    def check_undefined_var(self, should_warn):
        @jit
        def foo(result, n):
            if n:
                if n > 0:
                    c = 0
                result[0] = c
            else:
                # variable c is not defined in this branch
                c += 1
                result[0] = c

        if should_warn:
            with self.assertWarns(errors.NumbaWarning) as warns:
                # n=1 so we won't actually run the branch with the uninitialized
                self.check_func(foo, np.array([0]), 1)
            self.assertIn(
                "Detected uninitialized variable c", str(warns.warning)
            )
        else:
            self.check_func(foo, np.array([0]), 1)

        with self.assertRaises(UnboundLocalError):
            result = np.array([0])
            foo.py_func(result, 0)

    @skip_on_cudasim(
        "Numba variable warnings are not supported in the simulator"
    )
    def test_undefined_var(self):
        with override_config("ALWAYS_WARN_UNINIT_VAR", 0):
            self.check_undefined_var(should_warn=False)
        with override_config("ALWAYS_WARN_UNINIT_VAR", 1):
            self.check_undefined_var(should_warn=True)

    def test_phi_propagation(self):
        @jit
        def foo(result, actions):
            n = 1

            i = 0
            ct = 0
            while n > 0 and i < len(actions):
                n -= 1

                while actions[i]:
                    if actions[i]:
                        if actions[i]:
                            n += 10
                        actions[i] -= 1
                    else:
                        if actions[i]:
                            n += 20
                        actions[i] += 1

                    ct += n
                ct += n
            result[0] = ct
            result[1] = n

        self.check_func(foo, np.array([1, 2]), np.array([1, 2]))


class TestReportedSSAIssues(SSABaseTest):
    # Tests from issues
    # https://github.com/numba/numba/issues?q=is%3Aopen+is%3Aissue+label%3ASSA

    def test_issue2194(self):
        @jit
        def foo(result, V):
            s = np.uint32(1)

            for i in range(s):
                V[i] = 1
            for i in range(s, 1):
                pass
            result[0] = V[0]

        V = np.empty(1)
        self.check_func(foo, np.array([1.0]), V)

    def test_issue3094(self):
        @jit
        def foo(result, pred):
            if pred:
                x = 1
            else:
                x = 0
            result[0] = x

        self.check_func(foo, np.array([0]), False)

    def test_issue3931(self):
        @jit
        def foo(result, arr):
            for i in range(1):
                arr = arr.reshape(3 * 2)
                arr = arr.reshape(3, 2)
            # Copy result array elements
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    result[i, j] = arr[i, j]

        result_gpu = np.zeros((3, 2))
        self.check_func(foo, result_gpu, np.zeros((3, 2)))

    def test_issue3976(self):
        def overload_this(a):
            return 42

        @jit
        def foo(result, a):
            if a:
                s = 5
                s = overload_this(s)
            else:
                s = 99

            result[0] = s

        @overload(overload_this)
        def ol(a):
            return overload_this

        self.check_func(foo, np.array([42]), True)

    def test_issue3979(self):
        @jit
        def foo(result, A, B):
            x = A[0]
            y = B[0]
            for i in A:
                x = i
            for i in B:
                y = i
            result[0] = x
            result[1] = y

        self.check_func(
            foo, np.array([2, 4]), np.array([1, 2]), np.array([3, 4])
        )

    def test_issue5219(self):
        def overload_this(a, b=None):
            if isinstance(b, tuple):
                b = b[0]
            return b

        @overload(overload_this)
        def ol(a, b=None):
            b_is_tuple = isinstance(b, (types.Tuple, types.UniTuple))

            def impl(a, b=None):
                if b_is_tuple is True:
                    b = b[0]
                return b

            return impl

        @jit
        def test_tuple(result, a, b):
            result[0] = overload_this(a, b)

        self.check_func(test_tuple, np.array([2]), 1, (2,))

    def test_issue5223(self):
        @jit
        def bar(result, x):
            if len(x) == 5:
                for i in range(len(x)):
                    result[i] = x[i]
            else:
                # Manual copy since .copy() not available in CUDA
                for i in range(len(x)):
                    result[i] = x[i] + 1

        a = np.ones(5)
        a.flags.writeable = False
        expected = np.ones(5)  # Since len(a) == 5, it should return unchanged
        self.check_func(bar, expected, a)

    def test_issue5243(self):
        @jit
        def foo(result, q, lin):
            stencil_val = 0.0  # noqa: F841
            stencil_val = q[0, 0]  # noqa: F841
            result[0] = lin[0]

        lin = np.array([0.1, 0.6, 0.3])
        self.check_func(foo, np.array([0.1]), np.zeros((2, 2)), lin)

    def test_issue5482_missing_variable_init(self):
        # Test error that lowering fails because variable is missing
        # a definition before use.
        @jit
        def foo(result, x, v, n):
            problematic = 0  # Initialize to avoid unbound variable
            for i in range(n):
                if i == 0:
                    if i == x:
                        pass
                    else:
                        problematic = v
                else:
                    if i == x:
                        pass
                    else:
                        problematic = problematic + v
            result[0] = problematic

        self.check_func(foo, np.array([10]), 1, 5, 3)

    def test_issue5493_unneeded_phi(self):
        # Test error that unneeded phi is inserted because variable does not
        # have a dominance definition.
        data = (np.ones(2), np.ones(2))
        A = np.ones(1)
        B = np.ones(1)

        @jit
        def foo(res, m, n, data):
            if len(data) == 1:
                v0 = data[0]
            else:
                v0 = data[0]
                # Unneeded PHI node for `problematic` would be placed here
                for _ in range(1, len(data)):
                    v0[0] += A[0]

            for t in range(1, m):
                for idx in range(n):
                    t = B

                    if idx == 0:
                        if idx == n - 1:
                            pass
                        else:
                            res[0] = t[0]
                    else:
                        if idx == n - 1:
                            pass
                        else:
                            res[0] += t[0]

        self.check_func(foo, np.array([10]), 10, 10, data)

    def test_issue5623_equal_statements_in_same_bb(self):
        def foo(pred, stack):
            i = 0
            c = 1

            if pred is True:
                stack[i] = c
                i += 1
                stack[i] = c
                i += 1

        python = np.array([0, 666])
        foo(True, python)

        nb = np.array([0, 666])

        # Convert to CUDA kernel
        foo_cuda = jit(foo)
        foo_cuda[1, 1](True, nb)

        expect = np.array([1, 1])

        np.testing.assert_array_equal(python, expect)
        np.testing.assert_array_equal(nb, expect)
