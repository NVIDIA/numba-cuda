import functools
import sys
import warnings

import numpy as np

import unittest
from numba import njit, typeof
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin
from numba.np import numpy_support

is32bits = tuple.__itemsize__ == 4
iswindows = sys.platform.startswith("win32")


def _unimplemented(func):
    """An 'expectedFailure' like decorator that only expects compilation errors
    caused by unimplemented functions that fail in no-python mode"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypingError:
            raise unittest._ExpectedFailure(sys.exc_info())
        raise unittest._UnexpectedSuccess


def _make_ufunc_usecase(ufunc):
    ldict = {}
    arg_str = ",".join(["a{0}".format(i) for i in range(ufunc.nargs)])
    func_str = "def fn({0}):\n    np.{1}({0})".format(arg_str, ufunc.__name__)
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "{0}_usecase".format(ufunc.__name__)
    return fn


def _make_unary_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec("def fn(x):\n    return {0}(x)".format(ufunc_op), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    return fn


def _make_binary_ufunc_op_usecase(ufunc_op):
    ldict = {}
    exec("def fn(x,y):\n    return x{0}y".format(ufunc_op), globals(), ldict)
    fn = ldict["fn"]
    fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    return fn


def _make_inplace_ufunc_op_usecase(ufunc_op):
    """Generates a function to be compiled that performs an inplace operation

    ufunc_op can be a string like '+=' or a function like operator.iadd
    """
    if isinstance(ufunc_op, str):
        ldict = {}
        exec("def fn(x,y):\n    x{0}y".format(ufunc_op), globals(), ldict)
        fn = ldict["fn"]
        fn.__name__ = "usecase_{0}".format(hash(ufunc_op))
    else:

        def inplace_op(x, y):
            ufunc_op(x, y)

        fn = inplace_op
    return fn


def _as_dtype_value(tyargs, args):
    """Convert python values into numpy scalar objects."""
    return [np.dtype(str(ty)).type(val) for ty, val in zip(tyargs, args)]


class BaseUFuncTest(MemoryLeakMixin):
    def setUp(self):
        super(BaseUFuncTest, self).setUp()
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
