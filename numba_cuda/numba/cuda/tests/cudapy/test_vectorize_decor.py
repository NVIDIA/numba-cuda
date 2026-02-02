# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import math

from numba.cuda import vectorize, int32, uint32, float32, float64
from numba.cuda import config
from numba.cuda.testing import (
    skip_on_cudasim,
    CUDATestCase,
    skip_if_cupy_unavailable,
)
from numba.cuda.tests.support import CheckWarningsMixin

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    import cupy as cp
import unittest


pi = math.pi


def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return math.sin(x * pi) / (pi * x)


def scaled_sinc(x, scale):
    if x == 0.0:
        return scale
    else:
        return scale * (math.sin(x * pi) / (pi * x))


def vector_add(a, b):
    return a + b


class BaseVectorizeDecor:
    target = None
    wrapper = None
    funcs = {
        "func1": sinc,
        "func2": scaled_sinc,
        "func3": vector_add,
    }

    @classmethod
    def _run_and_compare(cls, func, sig, A, *args, **kwargs):
        if cls.wrapper is not None:
            func = cls.wrapper(func)
        numba_func = vectorize(sig, target=cls.target)(func)
        numpy_func = np.vectorize(func)
        result = numba_func(A, *args)
        gold = numpy_func(A, *args)
        np.testing.assert_allclose(result, gold, **kwargs)

    def test_1(self):
        sig = ["float64(float64)", "float32(float32)"]
        func = self.funcs["func1"]
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_2(self):
        sig = [float64(float64), float32(float32)]
        func = self.funcs["func1"]
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A)

    def test_3(self):
        sig = ["float64(float64, uint32)"]
        func = self.funcs["func2"]
        A = np.arange(100, dtype=np.float64)
        scale = np.uint32(3)
        self._run_and_compare(func, sig, A, scale, atol=1e-8)

    def test_4(self):
        sig = [
            int32(int32, int32),
            uint32(uint32, uint32),
            float32(float32, float32),
            float64(float64, float64),
        ]
        func = self.funcs["func3"]
        A = np.arange(100, dtype=np.float64)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.float32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.int32)
        self._run_and_compare(func, sig, A, A)
        A = A.astype(np.uint32)
        self._run_and_compare(func, sig, A, A)


class BaseVectorizeNopythonArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing the nopython argument to the vectorize decorator.
    """

    def _test_target_nopython(self, target, warnings, with_sig=True):
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.check_warnings(warnings):
            f = vectorize(*args, target=target, nopython=True)(vector_add)
            f(a, b)


class BaseVectorizeUnrecognizedArg(unittest.TestCase, CheckWarningsMixin):
    """
    Test passing an unrecognized argument to the vectorize decorator.
    """

    def _test_target_unrecognized_arg(self, target, with_sig=True):
        a = np.array([2.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        sig = [float32(float32, float32)]
        args = with_sig and [sig] or []
        with self.assertRaises(KeyError) as raises:
            f = vectorize(*args, target=target, nonexistent=2)(vector_add)
            f(a, b)
        self.assertIn("Unrecognized options", str(raises.exception))


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestVectorizeDecor(CUDATestCase, BaseVectorizeDecor):
    """
    Runs the tests from BaseVectorizeDecor with the CUDA target.
    """

    target = "cuda"


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestGPUVectorizeBroadcast(CUDATestCase):
    def test_broadcast(self):
        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(["float64(float64,float64)"], target="cuda")
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(a, b)
        np.testing.assert_almost_equal(expect, got)

    @skip_if_cupy_unavailable
    def test_device_broadcast(self):
        """
        Same test as .test_broadcast() but with device array as inputs
        """

        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(["float64(float64,float64)"], target="cuda")
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(cp.asarray(a), cp.asarray(b))
        np.testing.assert_almost_equal(expect, got.copy_to_host())


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestVectorizeNopythonArg(BaseVectorizeNopythonArg, CUDATestCase):
    def test_target_cuda_nopython(self):
        warnings = ["nopython kwarg for cuda target is redundant"]
        self._test_target_nopython("cuda", warnings)


@skip_on_cudasim("ufunc API unsupported in the simulator")
class TestVectorizeUnrecognizedArg(BaseVectorizeUnrecognizedArg, CUDATestCase):
    def test_target_cuda_unrecognized_arg(self):
        self._test_target_unrecognized_arg("cuda")


if __name__ == "__main__":
    unittest.main()
