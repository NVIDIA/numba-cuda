# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import itertools

from numba.cuda.core import errors
from numba.cuda import types, typing
from numba.cuda.typeconv import Conversion

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.cudapy.test_typeconv import CompatibilityTestMixin
from numba.cuda.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.cuda.core.typed_passes import PartialTypeInference
from numba.cuda.core.compiler_machinery import FunctionPass, register_pass
import unittest

from numba.cuda.flags import Flags


i8 = types.int8
i16 = types.int16
i32 = types.int32
i64 = types.int64
u8 = types.uint8
u16 = types.uint16
u32 = types.uint32
u64 = types.uint64
f32 = types.float32
f64 = types.float64
c64 = types.complex64
c128 = types.complex128


class TestUnify(CUDATestCase):
    """
    Tests for type unification with a typing context.
    """

    int_unify = {
        ("uint8", "uint8"): "uint8",
        ("int8", "int8"): "int8",
        ("uint16", "uint16"): "uint16",
        ("int16", "int16"): "int16",
        ("uint32", "uint32"): "uint32",
        ("int32", "int32"): "int32",
        ("uint64", "uint64"): "uint64",
        ("int64", "int64"): "int64",
        ("int8", "uint8"): "int16",
        ("int8", "uint16"): "int32",
        ("int8", "uint32"): "int64",
        ("uint8", "int32"): "int32",
        ("uint8", "uint64"): "uint64",
        ("int16", "int8"): "int16",
        ("int16", "uint8"): "int16",
        ("int16", "uint16"): "int32",
        ("int16", "uint32"): "int64",
        ("int16", "int64"): "int64",
        ("int16", "uint64"): "float64",
        ("uint16", "uint8"): "uint16",
        ("uint16", "uint32"): "uint32",
        ("uint16", "int32"): "int32",
        ("uint16", "uint64"): "uint64",
        ("int32", "int8"): "int32",
        ("int32", "int16"): "int32",
        ("int32", "uint32"): "int64",
        ("int32", "int64"): "int64",
        ("uint32", "uint8"): "uint32",
        ("uint32", "int64"): "int64",
        ("uint32", "uint64"): "uint64",
        ("int64", "int8"): "int64",
        ("int64", "uint8"): "int64",
        ("int64", "uint16"): "int64",
        ("uint64", "int8"): "float64",
        ("uint64", "int32"): "float64",
        ("uint64", "int64"): "float64",
    }

    def assert_unify(self, aty, bty, expected):
        ctx = typing.Context()
        template = "{0}, {1} -> {2} != {3}"
        for unify_func in ctx.unify_types, ctx.unify_pairs:
            unified = unify_func(aty, bty)
            self.assertEqual(
                unified,
                expected,
                msg=template.format(aty, bty, unified, expected),
            )
            unified = unify_func(bty, aty)
            self.assertEqual(
                unified,
                expected,
                msg=template.format(bty, aty, unified, expected),
            )

    def assert_unify_failure(self, aty, bty):
        self.assert_unify(aty, bty, None)

    def test_integer(self):
        ctx = typing.Context()  # noqa: F841
        for aty, bty in itertools.product(
            types.integer_domain, types.integer_domain
        ):
            key = (str(aty), str(bty))
            try:
                expected = self.int_unify[key]
            except KeyError:
                expected = self.int_unify[key[::-1]]
            self.assert_unify(aty, bty, getattr(types, expected))

    def test_bool(self):
        aty = types.boolean
        for bty in types.integer_domain:
            self.assert_unify(aty, bty, bty)
        # Not sure about this one, but it respects transitivity
        for cty in types.real_domain:
            self.assert_unify(aty, cty, cty)

    def unify_number_pair_test(self, n):
        """
        Test all permutations of N-combinations of numeric types and ensure
        that the order of types in the sequence is irrelevant.
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, n):
            res = [
                ctx.unify_types(*comb) for comb in itertools.permutations(tys)
            ]
            first_result = res[0]
            # Sanity check
            self.assertIsInstance(first_result, types.Number)
            # All results must be equal
            for other in res[1:]:
                self.assertEqual(first_result, other)

    def test_unify_number_pair(self):
        self.unify_number_pair_test(2)
        self.unify_number_pair_test(3)

    def test_none_to_optional(self):
        """
        Test unification of `none` and multiple number types to optional type
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, 2):
            # First unify without none, to provide the control value
            tys = list(tys)
            expected = types.Optional(ctx.unify_types(*tys))
            results = [
                ctx.unify_types(*comb)
                for comb in itertools.permutations(tys + [types.none])
            ]
            # All results must be equal
            for res in results:
                self.assertEqual(res, expected)

    def test_none(self):
        aty = types.none
        bty = types.none
        self.assert_unify(aty, bty, types.none)

    def test_optional(self):
        aty = types.Optional(i32)
        bty = types.none
        self.assert_unify(aty, bty, aty)
        aty = types.Optional(i32)
        bty = types.Optional(i64)
        self.assert_unify(aty, bty, bty)
        aty = types.Optional(i32)
        bty = i64
        self.assert_unify(aty, bty, types.Optional(i64))
        # Failure
        aty = types.Optional(i32)
        bty = types.Optional(types.slice3_type)
        self.assert_unify_failure(aty, bty)

    def test_tuple(self):
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(i64, 3)
        self.assert_unify(aty, bty, types.UniTuple(i64, 3))
        # (Tuple, UniTuple) -> Tuple
        aty = types.UniTuple(i32, 2)
        bty = types.Tuple((i16, i64))
        self.assert_unify(aty, bty, types.Tuple((i32, i64)))
        aty = types.UniTuple(i64, 0)
        bty = types.Tuple(())
        self.assert_unify(aty, bty, bty)
        # (Tuple, Tuple) -> Tuple
        aty = types.Tuple((i8, i16, i32))
        bty = types.Tuple((i32, i16, i8))
        self.assert_unify(aty, bty, types.Tuple((i32, i16, i32)))
        aty = types.Tuple((i8, i32))
        bty = types.Tuple((i32, i8))
        self.assert_unify(aty, bty, types.Tuple((i32, i32)))
        aty = types.Tuple((i8, i16))
        bty = types.Tuple((i16, i8))
        self.assert_unify(aty, bty, types.Tuple((i16, i16)))
        # Different number kinds
        aty = types.UniTuple(f64, 3)
        bty = types.UniTuple(c64, 3)
        self.assert_unify(aty, bty, types.UniTuple(c128, 3))
        # Tuples of tuples
        aty = types.UniTuple(types.Tuple((u32, f32)), 2)
        bty = types.UniTuple(types.Tuple((i16, f32)), 2)
        self.assert_unify(aty, bty, types.UniTuple(types.Tuple((i64, f32)), 2))
        # Failures
        aty = types.UniTuple(i32, 1)
        bty = types.UniTuple(types.slice3_type, 1)
        self.assert_unify_failure(aty, bty)
        aty = types.UniTuple(i32, 1)
        bty = types.UniTuple(i32, 2)
        self.assert_unify_failure(aty, bty)
        aty = types.Tuple((i8, types.slice3_type))
        bty = types.Tuple((i32, i8))
        self.assert_unify_failure(aty, bty)

    def test_optional_tuple(self):
        # Unify to optional tuple
        aty = types.none
        bty = types.UniTuple(i32, 2)
        self.assert_unify(aty, bty, types.Optional(types.UniTuple(i32, 2)))
        aty = types.Optional(types.UniTuple(i16, 2))
        bty = types.UniTuple(i32, 2)
        self.assert_unify(aty, bty, types.Optional(types.UniTuple(i32, 2)))
        # Unify to tuple of optionals
        aty = types.Tuple((types.none, i32))
        bty = types.Tuple((i16, types.none))
        self.assert_unify(
            aty, bty, types.Tuple((types.Optional(i16), types.Optional(i32)))
        )
        aty = types.Tuple((types.Optional(i32), i64))
        bty = types.Tuple((i16, types.Optional(i8)))
        self.assert_unify(
            aty, bty, types.Tuple((types.Optional(i32), types.Optional(i64)))
        )

    def test_arrays(self):
        aty = types.Array(i32, 3, "C")
        bty = types.Array(i32, 3, "A")
        self.assert_unify(aty, bty, bty)
        aty = types.Array(i32, 3, "C")
        bty = types.Array(i32, 3, "F")
        self.assert_unify(aty, bty, types.Array(i32, 3, "A"))
        aty = types.Array(i32, 3, "C")
        bty = types.Array(i32, 3, "C", readonly=True)
        self.assert_unify(aty, bty, bty)
        aty = types.Array(i32, 3, "A")
        bty = types.Array(i32, 3, "C", readonly=True)
        self.assert_unify(aty, bty, types.Array(i32, 3, "A", readonly=True))
        # Failures
        aty = types.Array(i32, 2, "C")
        bty = types.Array(i32, 3, "C")
        self.assert_unify_failure(aty, bty)
        aty = types.Array(i32, 2, "C")
        bty = types.Array(u32, 2, "C")
        self.assert_unify_failure(aty, bty)

    def test_list(self):
        aty = types.List(types.undefined)
        bty = types.List(i32)
        self.assert_unify(aty, bty, bty)
        aty = types.List(i16)
        bty = types.List(i32)
        self.assert_unify(aty, bty, bty)
        aty = types.List(types.Tuple([i32, i16]))
        bty = types.List(types.Tuple([i16, i64]))
        cty = types.List(types.Tuple([i32, i64]))
        self.assert_unify(aty, bty, cty)
        # Different reflections
        aty = types.List(i16, reflected=True)
        bty = types.List(i32)
        cty = types.List(i32, reflected=True)
        self.assert_unify(aty, bty, cty)
        # Incompatible dtypes
        aty = types.List(i16)
        bty = types.List(types.Tuple([i16]))
        self.assert_unify_failure(aty, bty)

    def test_set(self):
        # Different reflections
        aty = types.Set(i16, reflected=True)
        bty = types.Set(i32)
        cty = types.Set(i32, reflected=True)
        self.assert_unify(aty, bty, cty)
        # Incompatible dtypes
        aty = types.Set(i16)
        bty = types.Set(types.Tuple([i16]))
        self.assert_unify_failure(aty, bty)

    def test_range(self):
        aty = types.range_state32_type
        bty = types.range_state64_type
        self.assert_unify(aty, bty, bty)


class TestTypeConversion(CompatibilityTestMixin, unittest.TestCase):
    """
    Test for conversion between types with a typing context.
    """

    def assert_can_convert(self, aty, bty, expected):
        ctx = typing.Context()
        got = ctx.can_convert(aty, bty)
        self.assertEqual(got, expected)

    def assert_cannot_convert(self, aty, bty):
        ctx = typing.Context()
        got = ctx.can_convert(aty, bty)
        self.assertIsNone(got)

    def test_convert_number_types(self):
        # Check that Context.can_convert() is compatible with the default
        # number conversion rules registered in the typeconv module
        # (which is used internally by the C _Dispatcher object).
        ctx = typing.Context()
        self.check_number_compatibility(ctx.can_convert)

    def test_tuple(self):
        # UniTuple -> UniTuple
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(i64, 3)
        self.assert_can_convert(aty, aty, Conversion.exact)
        self.assert_can_convert(aty, bty, Conversion.promote)
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(f64, 3)
        self.assert_can_convert(aty, bty, Conversion.safe)
        # Tuple -> Tuple
        aty = types.Tuple((i32, i32))
        bty = types.Tuple((i32, i64))
        self.assert_can_convert(aty, bty, Conversion.promote)
        # UniTuple <-> Tuple
        aty = types.UniTuple(i32, 2)
        bty = types.Tuple((i32, i64))
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_can_convert(bty, aty, Conversion.unsafe)
        # Empty tuples
        aty = types.UniTuple(i64, 0)
        bty = types.UniTuple(i32, 0)
        cty = types.Tuple(())
        self.assert_can_convert(aty, bty, Conversion.safe)
        self.assert_can_convert(bty, aty, Conversion.safe)
        self.assert_can_convert(aty, cty, Conversion.safe)
        self.assert_can_convert(cty, aty, Conversion.safe)
        # Failures
        aty = types.UniTuple(i64, 3)
        bty = types.UniTuple(types.none, 3)
        self.assert_cannot_convert(aty, bty)
        aty = types.UniTuple(i64, 2)
        bty = types.UniTuple(i64, 3)

    def test_arrays(self):
        # Different layouts
        aty = types.Array(i32, 3, "C")
        bty = types.Array(i32, 3, "A")
        self.assert_can_convert(aty, bty, Conversion.safe)
        aty = types.Array(i32, 2, "C")
        bty = types.Array(i32, 2, "F")
        self.assert_cannot_convert(aty, bty)
        # Different mutabilities
        aty = types.Array(i32, 3, "C")
        bty = types.Array(i32, 3, "C", readonly=True)
        self.assert_can_convert(aty, aty, Conversion.exact)
        self.assert_can_convert(bty, bty, Conversion.exact)
        self.assert_can_convert(aty, bty, Conversion.safe)
        self.assert_cannot_convert(bty, aty)
        # Various failures
        aty = types.Array(i32, 2, "C")
        bty = types.Array(i32, 3, "C")
        self.assert_cannot_convert(aty, bty)
        aty = types.Array(i32, 2, "C")
        bty = types.Array(i64, 2, "C")
        self.assert_cannot_convert(aty, bty)

    def test_optional(self):
        aty = types.int32
        bty = types.Optional(i32)
        self.assert_can_convert(types.none, bty, Conversion.promote)
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_cannot_convert(bty, types.none)
        self.assert_can_convert(bty, aty, Conversion.safe)  # XXX ???
        # Optional array
        aty = types.Array(i32, 2, "C")
        bty = types.Optional(aty)
        self.assert_can_convert(types.none, bty, Conversion.promote)
        self.assert_can_convert(aty, bty, Conversion.promote)
        self.assert_can_convert(bty, aty, Conversion.safe)
        aty = types.Array(i32, 2, "C")
        bty = types.Optional(aty.copy(layout="A"))
        self.assert_can_convert(aty, bty, Conversion.safe)  # C -> A
        self.assert_cannot_convert(bty, aty)  # A -> C
        aty = types.Array(i32, 2, "C")
        bty = types.Optional(aty.copy(layout="F"))
        self.assert_cannot_convert(aty, bty)
        self.assert_cannot_convert(bty, aty)


class TestResolveOverload(unittest.TestCase):
    """
    Tests for typing.Context.resolve_overload().
    """

    def assert_resolve_overload(self, cases, args, expected):
        ctx = typing.Context()
        got = ctx.resolve_overload("foo", cases, args, {})
        self.assertEqual(got, expected)

    def test_non_ambiguous_match(self):
        def check(args, expected):
            self.assert_resolve_overload(cases, args, expected)
            # Order shouldn't matter here
            self.assert_resolve_overload(cases[::-1], args, expected)

        cases = [i8(i8, i8), i32(i32, i32), f64(f64, f64)]
        # Exact match
        check((i8, i8), cases[0])
        check((i32, i32), cases[1])
        check((f64, f64), cases[2])
        # "Promote" conversion
        check((i8, i16), cases[1])
        check((i32, i8), cases[1])
        check((i32, i8), cases[1])
        check((f32, f32), cases[2])
        # "Safe" conversion
        check((u32, u32), cases[2])
        # "Unsafe" conversion
        check((i64, i64), cases[2])

    def test_ambiguous_match(self):
        # When the best match is ambiguous (there is a tie), the first
        # best case in original sequence order should be returned.
        def check(args, expected, expected_reverse):
            self.assert_resolve_overload(cases, args, expected)
            self.assert_resolve_overload(cases[::-1], args, expected_reverse)

        cases = [i16(i16, i16), i32(i32, i32), f64(f64, f64)]
        # Two "promote" conversions
        check((i8, i8), cases[0], cases[1])
        # Two "safe" conversions
        check((u16, u16), cases[1], cases[2])

        cases = [i32(i32, i32), f32(f32, f32)]
        # Two "unsafe" conversions
        check((u32, u32), cases[0], cases[1])

    def test_ambiguous_error(self):
        ctx = typing.Context()
        cases = [i16(i16, i16), i32(i32, i32)]
        with self.assertRaises(TypeError) as raises:
            ctx.resolve_overload(
                "foo", cases, (i8, i8), {}, allow_ambiguous=False
            )
        self.assertEqual(
            str(raises.exception).splitlines(),
            [
                "Ambiguous overloading for foo (int8, int8):",
                "(int16, int16) -> int16",
                "(int32, int32) -> int32",
            ],
        )


@register_pass(mutates_CFG=False, analysis_only=True)
class DummyCR(FunctionPass):
    """Dummy pass to add "cr" to compiler state to avoid errors in TyperCompiler since
    it doesn't have lowering.
    """

    _name = "dummy_cr"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.cr = 1  # arbitrary non-None value
        return True


def get_func_typing_errs(func, arg_types):
    """
    Get typing errors for function 'func'. It creates a pipeline that runs untyped
    passes as well as type inference.
    """
    from numba.cuda.compiler import CompilerBase, PassManager

    class TyperCompiler(CompilerBase):
        """A compiler pipeline that skips passes after typing (provides partial typing info
        but not lowering).
        """

        def define_pipelines(self):
            pm = PassManager("custom_pipeline")
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(IRProcessing, "processing IR")
            pm.add_pass(PartialTypeInference, "do partial typing")
            pm.add_pass_after(DummyCR, PartialTypeInference)
            pm.finalize()
            return [pm]

    from numba.cuda.descriptor import cuda_target

    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context
    library = None
    return_type = None
    _locals = {}
    flags = Flags()

    pipeline = TyperCompiler(
        typingctx, targetctx, library, arg_types, return_type, flags, _locals
    )
    pipeline.compile_extra(func)
    return pipeline.state.typing_errors


@skip_on_cudasim
class TestPartialTypingErrors(CUDATestCase):
    """
    Make sure partial typing stores type errors in compiler state properly
    """

    def test_partial_typing_error(self):
        # example with type unification error
        def impl(flag):
            if flag:
                a = 1
            else:
                a = ""
            return a

        self.assertRaisesRegex(
            errors.TypingError,
            r"Cannot unify Literal\[int]\(1\) and Literal\[str]\(\) for 'a'",
            get_func_typing_errs,
            impl,
            (types.bool_,),
        )


if __name__ == "__main__":
    unittest.main()
