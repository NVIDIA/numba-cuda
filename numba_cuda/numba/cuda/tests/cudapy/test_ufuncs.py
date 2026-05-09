# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools
import itertools
import warnings
import numpy as np
import pytest
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba import njit
from numba.cuda import config, types
from numba.cuda.testing import skip_on_standalone_numba_cuda
from numba.cuda.typing.typeof import typeof
from numba.cuda.np import numpy_support
from numba.cuda.tests.support import reset_module_warnings


@pytest.fixture
def base_input():
    return [
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


@pytest.fixture(name="inputs")
def ufunc_setup(base_input):
    # The basic ufunc test does not set up complex inputs, so we'll add
    # some here for testing with CUDA.
    extra_input = [
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

    # Test with multiple dimensions
    extra_input += [
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

    # Add tests for other integer types
    extra_input += [
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

    inputs = base_input + extra_input

    low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
    warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY

    # Disable warnings about low gpu utilization in the test suite
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    # Disable warnings about host arrays in the test suite
    config.CUDA_WARN_ON_IMPLICIT_COPY = 0

    yield inputs

    config.CUDA_LOW_OCCUPANCY_WARNINGS = low_occupancy_warnings
    config.CUDA_WARN_ON_IMPLICIT_COPY = warn_on_implicit_copy


def basic_int_ufunc_test(name, inputs):
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
    basic_ufunc_test(name, inputs, skip_inputs=skip_inputs)


def signed_unsigned_cmp_test(comparison_ufunc, inputs):
    basic_ufunc_test(comparison_ufunc, inputs)

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

    pyfunc = _make_ufunc_usecase(comparison_ufunc)

    for a, b in additional_inputs:
        input_types = (typeof(a), typeof(b))
        output_type = types.Array(types.bool_, 1, "C")
        argtys = input_types + (output_type,)
        cfunc = _compile(pyfunc, argtys)

        if isinstance(a, np.ndarray):
            result = np.zeros(a.shape, dtype=np.bool_)
        else:
            result = np.zeros(1, dtype=np.bool_)

        expected = np.zeros_like(result)

        pyfunc(a, b, expected)
        cfunc(a, b, result)
        np.testing.assert_equal(expected, result)


def basic_ufunc_test(
    ufunc,
    inputs,
    skip_inputs=(),
    additional_inputs=(),
    int_output_type=None,
    float_output_type=None,
    kinds="ifc",
    positive_only=False,
):
    # Necessary to avoid some Numpy warnings being silenced, despite
    # the simplefilter() call below.
    reset_module_warnings(__name__)

    pyfunc = _make_ufunc_usecase(ufunc)

    for input_operand, input_type in itertools.chain(inputs, additional_inputs):
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

        output_type = _determine_output_type(
            input_type, int_output_type, float_output_type
        )

        input_types = (input_type,) * ufunc.nin
        output_types = (output_type,) * ufunc.nout
        argtys = input_types + output_types
        cfunc = _compile(pyfunc, argtys)

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
                np.zeros(1, dtype=out_ty.dtype.name) for out_ty in output_types
            ]
            expected = [
                np.zeros(1, dtype=out_ty.dtype.name) for out_ty in output_types
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


def _determine_output_type(
    input_type, int_output_type=None, float_output_type=None
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


def _make_ufunc_usecase(ufunc):
    ldict = {}
    arg_str = ",".join(["a{0}".format(i) for i in range(ufunc.nargs)])
    func_str = f"def fn({arg_str}):\n    np.{ufunc.__name__}({arg_str})"
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "{0}_usecase".format(ufunc.__name__)
    return fn


@skip_on_standalone_numba_cuda
@functools.lru_cache(maxsize=None)
def _compile(pyfunc, args, nrt=False):
    # NOTE: to test the implementation of Numpy ufuncs, we disable
    # rewriting of array expressions.
    return njit(args, _nrt=nrt, no_rewrites=True)(pyfunc)


class TestBasicTrigUFuncs:
    def test_sin_ufunc(self, inputs):
        basic_ufunc_test(np.sin, inputs, kinds="cf")

    def test_cos_ufunc(self, inputs):
        basic_ufunc_test(np.cos, inputs, kinds="cf")

    def test_tan_ufunc(self, inputs):
        basic_ufunc_test(np.tan, inputs, kinds="cf")

    def test_arcsin_ufunc(self, inputs):
        basic_ufunc_test(np.arcsin, inputs, kinds="cf")

    def test_arccos_ufunc(self, inputs):
        basic_ufunc_test(np.arccos, inputs, kinds="cf")

    def test_arctan_ufunc(self, inputs):
        basic_ufunc_test(np.arctan, inputs, kinds="cf")

    def test_arctan2_ufunc(self, inputs):
        basic_ufunc_test(np.arctan2, inputs, kinds="f")


class TestHypTrigUFuncs:
    def test_hypot_ufunc(self, inputs):
        basic_ufunc_test(np.hypot, inputs, kinds="f")

    def test_sinh_ufunc(self, inputs):
        basic_ufunc_test(np.sinh, inputs, kinds="cf")

    def test_cosh_ufunc(self, inputs):
        basic_ufunc_test(np.cosh, inputs, kinds="cf")

    def test_tanh_ufunc(self, inputs):
        basic_ufunc_test(np.tanh, inputs, kinds="cf")

    def test_arcsinh_ufunc(self, inputs):
        basic_ufunc_test(np.arcsinh, inputs, kinds="cf")

    def test_arccosh_ufunc(self, inputs):
        basic_ufunc_test(np.arccosh, inputs, kinds="cf")

    def test_arctanh_ufunc(self, inputs):
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

        basic_ufunc_test(np.arctanh, inputs, skip_inputs=to_skip, kinds="cf")


class TestConversionUFuncs:
    def test_deg2rad_ufunc(self, inputs):
        basic_ufunc_test(np.deg2rad, inputs, kinds="f")

    def test_rad2deg_ufunc(self, inputs):
        basic_ufunc_test(np.rad2deg, inputs, kinds="f")

    def test_degrees_ufunc(self, inputs):
        basic_ufunc_test(np.degrees, inputs, kinds="f")

    def test_radians_ufunc(self, inputs):
        basic_ufunc_test(np.radians, inputs, kinds="f")


class TestComparisonUFuncs1:
    def test_greater_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.greater, inputs)

    def test_greater_equal_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.greater_equal, inputs)

    def test_less_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.less, inputs)

    def test_less_equal_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.less_equal, inputs)

    def test_not_equal_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.not_equal, inputs)

    def test_equal_ufunc(self, inputs):
        signed_unsigned_cmp_test(np.equal, inputs)


class TestLogicalUFuncs:
    def test_logical_and_ufunc(self, inputs):
        basic_ufunc_test(np.logical_and, inputs)

    def test_logical_or_ufunc(self, inputs):
        basic_ufunc_test(np.logical_or, inputs)

    def test_logical_xor_ufunc(self, inputs):
        basic_ufunc_test(np.logical_xor, inputs)

    def test_logical_not_ufunc(self, inputs):
        basic_ufunc_test(np.logical_not, inputs)


class TestMinmaxUFuncs:
    def test_maximum_ufunc(self, inputs):
        basic_ufunc_test(np.maximum, inputs)

    def test_minimum_ufunc(self, inputs):
        basic_ufunc_test(np.minimum, inputs)

    def test_fmax_ufunc(self, inputs):
        basic_ufunc_test(np.fmax, inputs)

    def test_fmin_ufunc(self, inputs):
        basic_ufunc_test(np.fmin, inputs)


class TestBitwiseUFuncs:
    def test_bitwise_and_ufunc(self, inputs):
        basic_int_ufunc_test(np.bitwise_and, inputs)

    def test_bitwise_or_ufunc(self, inputs):
        basic_int_ufunc_test(np.bitwise_or, inputs)

    def test_bitwise_xor_ufunc(self, inputs):
        basic_int_ufunc_test(np.bitwise_xor, inputs)

    def test_invert_ufunc(self, inputs):
        basic_int_ufunc_test(np.invert, inputs)

    def test_bitwise_not_ufunc(self, inputs):
        basic_int_ufunc_test(np.bitwise_not, inputs)


class TestLogUFuncs:
    def test_log_ufunc(self, inputs):
        basic_ufunc_test(np.log, inputs, kinds="cf")

    def test_log2_ufunc(self, inputs):
        basic_ufunc_test(np.log2, inputs, kinds="cf")

    def test_log10_ufunc(self, inputs):
        basic_ufunc_test(np.log10, inputs, kinds="cf")
