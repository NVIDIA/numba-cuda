# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools
import warnings
import numpy as np
import unittest
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba import njit
from numba import cuda
from numba.cuda import config, types
from numba.cuda.testing import skip_on_standalone_numba_cuda
from numba.cuda.typing.typeof import typeof
from numba.cuda.np import numpy_support
from numba.cuda.tests.support import TestCase


class BaseUFuncTest:
    def setUp(self):
        self.inputs = [
            (np.uint32(0), types.uint32),
            (np.uint32(1), types.uint32),
            (np.int32(-1), types.int32),
            (np.int32(0), types.int32),
            (np.int32(1), types.int32),
            (np.uint64(0), types.uint64),
            (np.uint64(1), types.uint64),
            (np.int64(-1), types.int64),
            (np.int64(0), types.int64),
            (np.int64(1), types.int64),
            (np.float32(-0.5), types.float32),
            (np.float32(0.0), types.float32),
            (np.float32(0.5), types.float32),
            (np.float64(-0.5), types.float64),
            (np.float64(0.0), types.float64),
            (np.float64(0.5), types.float64),
            (np.array([0, 1], dtype="u4"), types.Array(types.uint32, 1, "C")),
            (np.array([0, 1], dtype="u8"), types.Array(types.uint64, 1, "C")),
            (
                np.array([-1, 0, 1], dtype="i4"),
                types.Array(types.int32, 1, "C"),
            ),
            (
                np.array([-1, 0, 1], dtype="i8"),
                types.Array(types.int64, 1, "C"),
            ),
            (
                np.array([-0.5, 0.0, 0.5], dtype="f4"),
                types.Array(types.float32, 1, "C"),
            ),
            (
                np.array([-0.5, 0.0, 0.5], dtype="f8"),
                types.Array(types.float64, 1, "C"),
            ),
            (np.array([0, 1], dtype=np.int8), types.Array(types.int8, 1, "C")),
            (
                np.array([0, 1], dtype=np.int16),
                types.Array(types.int16, 1, "C"),
            ),
            (
                np.array([0, 1], dtype=np.uint8),
                types.Array(types.uint8, 1, "C"),
            ),
            (
                np.array([0, 1], dtype=np.uint16),
                types.Array(types.uint16, 1, "C"),
            ),
        ]

    @skip_on_standalone_numba_cuda
    @functools.lru_cache(maxsize=None)
    def _compile(self, pyfunc, args, nrt=False):
        # NOTE: to test the implementation of Numpy ufuncs, we disable
        # rewriting of array expressions.
        return njit(args, _nrt=nrt, no_rewrites=True)(pyfunc)

    def _determine_output_type(
        self, input_type, int_output_type=None, float_output_type=None
    ):
        ty = input_type
        if isinstance(ty, types.Array):
            ndim = ty.ndim
            ty = ty.dtype
        else:
            ndim = 1

        if ty in types.signed_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, ndim, "C")
            else:
                output_type = types.Array(ty, ndim, "C")
        elif ty in types.unsigned_domain:
            if int_output_type:
                output_type = types.Array(int_output_type, ndim, "C")
            else:
                output_type = types.Array(ty, ndim, "C")
        else:
            if float_output_type:
                output_type = types.Array(float_output_type, ndim, "C")
            else:
                output_type = types.Array(ty, ndim, "C")
        return output_type


class BasicUFuncTest(BaseUFuncTest):
    def _make_ufunc_usecase(self, ufunc):
        return _make_ufunc_usecase(ufunc)

    def basic_ufunc_test(
        self,
        ufunc,
        skip_inputs=[],
        additional_inputs=[],
        int_output_type=None,
        float_output_type=None,
        kinds="ifc",
        positive_only=False,
    ):
        # Necessary to avoid some Numpy warnings being silenced, despite
        # the simplefilter() call below.
        self.reset_module_warnings(__name__)

        pyfunc = self._make_ufunc_usecase(ufunc)

        inputs = list(self.inputs) + additional_inputs

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            is_tuple = isinstance(input_operand, tuple)
            if is_tuple:
                args = input_operand
            else:
                args = (input_operand,) * ufunc.nin

            if input_type in skip_inputs:
                continue
            if positive_only and np.any(args[0] < 0):
                continue

            # Some ufuncs don't allow all kinds of arguments
            if args[0].dtype.kind not in kinds:
                continue

            output_type = self._determine_output_type(
                input_type, int_output_type, float_output_type
            )

            input_types = (input_type,) * ufunc.nin
            output_types = (output_type,) * ufunc.nout
            argtys = input_types + output_types
            cfunc = self._compile(pyfunc, argtys)

            if isinstance(args[0], np.ndarray):
                results = [
                    np.zeros(args[0].shape, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
                expected = [
                    np.zeros(args[0].shape, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
            else:
                results = [
                    np.zeros(1, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
                expected = [
                    np.zeros(1, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]

            invalid_flag = False
            with warnings.catch_warnings(record=True) as warnlist:
                warnings.simplefilter("always")
                pyfunc(*args, *expected)

                warnmsg = "invalid value encountered"
                for thiswarn in warnlist:
                    if issubclass(thiswarn.category, RuntimeWarning) and str(
                        thiswarn.message
                    ).startswith(warnmsg):
                        invalid_flag = True

            cfunc(*args, *results)

            for expected_i, result_i in zip(expected, results):
                msg = "\n".join(
                    [
                        "ufunc '{0}' failed",
                        "inputs ({1}):",
                        "{2}",
                        "got({3})",
                        "{4}",
                        "expected ({5}):",
                        "{6}",
                    ]
                ).format(
                    ufunc.__name__,
                    input_type,
                    input_operand,
                    output_type,
                    result_i,
                    expected_i.dtype,
                    expected_i,
                )
                try:
                    np.testing.assert_array_almost_equal(
                        expected_i, result_i, decimal=5, err_msg=msg
                    )
                except AssertionError:
                    if invalid_flag:
                        # Allow output to mismatch for invalid input
                        print(
                            "Output mismatch for invalid input",
                            input_tuple,
                            result_i,
                            expected_i,
                        )
                    else:
                        raise

    def signed_unsigned_cmp_test(self, comparison_ufunc):
        self.basic_ufunc_test(comparison_ufunc)

        if numpy_support.numpy_version < (1, 25):
            return

        # Test additional implementations that specifically handle signed /
        # unsigned comparisons added in NumPy 1.25:
        # https://github.com/numpy/numpy/pull/23713
        additional_inputs = (
            (np.int64(-1), np.uint64(0)),
            (np.int64(-1), np.uint64(1)),
            (np.int64(0), np.uint64(0)),
            (np.int64(0), np.uint64(1)),
            (np.int64(1), np.uint64(0)),
            (np.int64(1), np.uint64(1)),
            (np.uint64(0), np.int64(-1)),
            (np.uint64(0), np.int64(0)),
            (np.uint64(0), np.int64(1)),
            (np.uint64(1), np.int64(-1)),
            (np.uint64(1), np.int64(0)),
            (np.uint64(1), np.int64(1)),
            (
                np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64),
                np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64),
            ),
            (
                np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64),
                np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64),
            ),
        )

        pyfunc = self._make_ufunc_usecase(comparison_ufunc)

        for a, b in additional_inputs:
            input_types = (typeof(a), typeof(b))
            output_type = types.Array(types.bool_, 1, "C")
            argtys = input_types + (output_type,)
            cfunc = self._compile(pyfunc, argtys)

            if isinstance(a, np.ndarray):
                result = np.zeros(a.shape, dtype=np.bool_)
            else:
                result = np.zeros(1, dtype=np.bool_)

            expected = np.zeros_like(result)

            pyfunc(a, b, expected)
            cfunc(a, b, result)
            np.testing.assert_equal(expected, result)


def _make_ufunc_usecase(ufunc):
    ldict = {}
    arg_str = ",".join(["a{0}".format(i) for i in range(ufunc.nargs)])
    func_str = f"def fn({arg_str}):\n    np.{ufunc.__name__}({arg_str})"
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "{0}_usecase".format(ufunc.__name__)
    return fn


# This class provides common functionality for UFunc tests. The UFunc tests
# are quite long-running in comparison to other tests, so we break the tests up
# into multiple test classes for distribution across workers.
#
# This class would also be a CUDATestCase, but to avoid a confusing and
# potentially dangerous inheritance diamond with setUp methods that modify
# global state, we implement the necessary part of CUDATestCase within this
# class instead. This disables CUDA performance warnings for the duration of
# tests.
class CUDAUFuncTestBase(BasicUFuncTest, TestCase):
    def setUp(self):
        BasicUFuncTest.setUp(self)

        # The basic ufunc test does not set up complex inputs, so we'll add
        # some here for testing with CUDA.
        self.inputs.extend(
            [
                (np.complex64(-0.5 - 0.5j), types.complex64),
                (np.complex64(0.0), types.complex64),
                (np.complex64(0.5 + 0.5j), types.complex64),
                (np.complex128(-0.5 - 0.5j), types.complex128),
                (np.complex128(0.0), types.complex128),
                (np.complex128(0.5 + 0.5j), types.complex128),
                (
                    np.array([-0.5 - 0.5j, 0.0, 0.5 + 0.5j], dtype="c8"),
                    types.Array(types.complex64, 1, "C"),
                ),
                (
                    np.array([-0.5 - 0.5j, 0.0, 0.5 + 0.5j], dtype="c16"),
                    types.Array(types.complex128, 1, "C"),
                ),
            ]
        )

        # Test with multiple dimensions
        self.inputs.extend(
            [
                # Basic 2D and 3D arrays
                (
                    np.linspace(0, 1).reshape((5, -1)),
                    types.Array(types.float64, 2, "C"),
                ),
                (
                    np.linspace(0, 1).reshape((2, 5, -1)),
                    types.Array(types.float64, 3, "C"),
                ),
                # Complex data (i.e. interleaved)
                (
                    np.linspace(0, 1 + 1j).reshape(5, -1),
                    types.Array(types.complex128, 2, "C"),
                ),
                # F-ordered
                (
                    np.asfortranarray(np.linspace(0, 1).reshape((5, -1))),
                    types.Array(types.float64, 2, "F"),
                ),
            ]
        )

        # Add tests for other integer types
        self.inputs.extend(
            [
                (np.uint8(0), types.uint8),
                (np.uint8(1), types.uint8),
                (np.int8(-1), types.int8),
                (np.int8(0), types.int8),
                (np.uint16(0), types.uint16),
                (np.uint16(1), types.uint16),
                (np.int16(-1), types.int16),
                (np.int16(0), types.int16),
                (np.ulonglong(0), types.ulonglong),
                (np.ulonglong(1), types.ulonglong),
                (np.longlong(-1), types.longlong),
                (np.longlong(0), types.longlong),
                (
                    np.array([0, 1], dtype=np.ulonglong),
                    types.Array(types.ulonglong, 1, "C"),
                ),
                (
                    np.array([0, 1], dtype=np.longlong),
                    types.Array(types.longlong, 1, "C"),
                ),
            ]
        )

        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY

        # Disable warnings about low gpu utilization in the test suite
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        # Disable warnings about host arrays in the test suite
        config.CUDA_WARN_ON_IMPLICIT_COPY = 0

    def tearDown(self):
        # Restore original warning settings
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy

    def _make_ufunc_usecase(self, ufunc):
        return _make_ufunc_usecase(ufunc)

    @functools.lru_cache(maxsize=None)
    def _compile(self, pyfunc, args):
        # We return an already-configured kernel so that basic_ufunc_test can
        # call it just like it does for a CPU function
        return cuda.jit(args)(pyfunc)[1, 1]

    def basic_int_ufunc_test(self, name=None):
        skip_inputs = [
            types.float32,
            types.float64,
            types.Array(types.float32, 1, "C"),
            types.Array(types.float32, 2, "C"),
            types.Array(types.float64, 1, "C"),
            types.Array(types.float64, 2, "C"),
            types.Array(types.float64, 3, "C"),
            types.Array(types.float64, 2, "F"),
            types.complex64,
            types.complex128,
            types.Array(types.complex64, 1, "C"),
            types.Array(types.complex64, 2, "C"),
            types.Array(types.complex128, 1, "C"),
            types.Array(types.complex128, 2, "C"),
        ]
        self.basic_ufunc_test(name, skip_inputs=skip_inputs)

    ############################################################################
    # Trigonometric Functions


class TestBasicTrigUFuncs(CUDAUFuncTestBase):
    def test_sin_ufunc(self):
        self.basic_ufunc_test(np.sin, kinds="cf")

    def test_cos_ufunc(self):
        self.basic_ufunc_test(np.cos, kinds="cf")

    def test_tan_ufunc(self):
        self.basic_ufunc_test(np.tan, kinds="cf")

    def test_arcsin_ufunc(self):
        self.basic_ufunc_test(np.arcsin, kinds="cf")

    def test_arccos_ufunc(self):
        self.basic_ufunc_test(np.arccos, kinds="cf")

    def test_arctan_ufunc(self):
        self.basic_ufunc_test(np.arctan, kinds="cf")

    def test_arctan2_ufunc(self):
        self.basic_ufunc_test(np.arctan2, kinds="f")


class TestHypTrigUFuncs(CUDAUFuncTestBase):
    def test_hypot_ufunc(self):
        self.basic_ufunc_test(np.hypot, kinds="f")

    def test_sinh_ufunc(self):
        self.basic_ufunc_test(np.sinh, kinds="cf")

    def test_cosh_ufunc(self):
        self.basic_ufunc_test(np.cosh, kinds="cf")

    def test_tanh_ufunc(self):
        self.basic_ufunc_test(np.tanh, kinds="cf")

    def test_arcsinh_ufunc(self):
        self.basic_ufunc_test(np.arcsinh, kinds="cf")

    def test_arccosh_ufunc(self):
        self.basic_ufunc_test(np.arccosh, kinds="cf")

    def test_arctanh_ufunc(self):
        # arctanh is only valid is only finite in the range ]-1, 1[
        # This means that for any of the integer types it will produce
        # conversion from infinity/-infinity to integer. That's undefined
        # behavior in C, so the results may vary from implementation to
        # implementation. This means that the result from the compiler
        # used to compile NumPy may differ from the result generated by
        # llvm. Skipping the integer types in this test avoids failed
        # tests because of this.
        to_skip = [
            types.Array(types.uint32, 1, "C"),
            types.uint32,
            types.Array(types.int32, 1, "C"),
            types.int32,
            types.Array(types.uint64, 1, "C"),
            types.uint64,
            types.Array(types.int64, 1, "C"),
            types.int64,
        ]

        self.basic_ufunc_test(np.arctanh, skip_inputs=to_skip, kinds="cf")


class TestConversionUFuncs(CUDAUFuncTestBase):
    def test_deg2rad_ufunc(self):
        self.basic_ufunc_test(np.deg2rad, kinds="f")

    def test_rad2deg_ufunc(self):
        self.basic_ufunc_test(np.rad2deg, kinds="f")

    def test_degrees_ufunc(self):
        self.basic_ufunc_test(np.degrees, kinds="f")

    def test_radians_ufunc(self):
        self.basic_ufunc_test(np.radians, kinds="f")

    ############################################################################
    # Comparison functions


class TestComparisonUFuncs1(CUDAUFuncTestBase):
    def test_greater_ufunc(self):
        self.signed_unsigned_cmp_test(np.greater)

    def test_greater_equal_ufunc(self):
        self.signed_unsigned_cmp_test(np.greater_equal)

    def test_less_ufunc(self):
        self.signed_unsigned_cmp_test(np.less)

    def test_less_equal_ufunc(self):
        self.signed_unsigned_cmp_test(np.less_equal)

    def test_not_equal_ufunc(self):
        self.signed_unsigned_cmp_test(np.not_equal)

    def test_equal_ufunc(self):
        self.signed_unsigned_cmp_test(np.equal)


class TestLogicalUFuncs(CUDAUFuncTestBase):
    def test_logical_and_ufunc(self):
        self.basic_ufunc_test(np.logical_and)

    def test_logical_or_ufunc(self):
        self.basic_ufunc_test(np.logical_or)

    def test_logical_xor_ufunc(self):
        self.basic_ufunc_test(np.logical_xor)

    def test_logical_not_ufunc(self):
        self.basic_ufunc_test(np.logical_not)


class TestMinmaxUFuncs(CUDAUFuncTestBase):
    def test_maximum_ufunc(self):
        self.basic_ufunc_test(np.maximum)

    def test_minimum_ufunc(self):
        self.basic_ufunc_test(np.minimum)

    def test_fmax_ufunc(self):
        self.basic_ufunc_test(np.fmax)

    def test_fmin_ufunc(self):
        self.basic_ufunc_test(np.fmin)


class TestBitwiseUFuncs(CUDAUFuncTestBase):
    def test_bitwise_and_ufunc(self):
        self.basic_int_ufunc_test(np.bitwise_and)

    def test_bitwise_or_ufunc(self):
        self.basic_int_ufunc_test(np.bitwise_or)

    def test_bitwise_xor_ufunc(self):
        self.basic_int_ufunc_test(np.bitwise_xor)

    def test_invert_ufunc(self):
        self.basic_int_ufunc_test(np.invert)

    def test_bitwise_not_ufunc(self):
        self.basic_int_ufunc_test(np.bitwise_not)

    # Note: there is no entry for np.left_shift and np.right_shift
    # because their implementations in NumPy have undefined behavior
    # when the second argument is a negative. See the comment in
    # numba/tests/test_ufuncs.py for more details.

    ############################################################################
    # Mathematical Functions


class TestLogUFuncs(CUDAUFuncTestBase):
    def test_log_ufunc(self):
        self.basic_ufunc_test(np.log, kinds="cf")

    def test_log2_ufunc(self):
        self.basic_ufunc_test(np.log2, kinds="cf")

    def test_log10_ufunc(self):
        self.basic_ufunc_test(np.log10, kinds="cf")


if __name__ == "__main__":
    unittest.main()
