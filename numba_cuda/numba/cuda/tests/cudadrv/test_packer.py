import unittest

from numba import types

from llvmlite import ir, binding as ll

from numba.cuda.descriptor import cuda_target
from numba.cuda.target import CUDACABIArgPacker

from numba.cuda.testing import CUDATestCase


class TestArgInfo(CUDATestCase):

    def _test_as_arguments(self, fe_args):
        """
        Test round-tripping types *fe_args* through the default data model's
        argument conversion and unpacking logic.
        """
        targetctx = cuda_target.target_context
        dmm = targetctx.data_model_manager
        fi = CUDACABIArgPacker(dmm, fe_args)

        module = ir.Module()
        fnty = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(module, fnty, name="test_arguments")
        builder = ir.IRBuilder()
        builder.position_at_end(function.append_basic_block())

        args = [ir.Constant(dmm.lookup(t).get_value_type(), None)
                for t in fe_args]

        # Roundtrip
        values = fi.as_arguments(builder, args)
        asargs = fi.from_arguments(builder, values)

        self.assertEqual(len(asargs), len(fe_args))
        valtys = tuple([v.type for v in values])
        self.assertEqual(valtys, fi.argument_types)

        expect_types = [a.type for a in args]
        got_types = [a.type for a in asargs]
        self.assertEqual(expect_types, got_types)

        # Test that values don't change representation when passed as
        # parameters
        self.assertEqual(valtys, tuple(a.type for a in args))

        # Assign names (check this doesn't raise)
        fi.assign_names(values, ["arg%i" for i in range(len(fe_args))])

        builder.ret_void()

        ll.parse_assembly(str(module))

    def test_int32_array_complex(self):
        fe_args = [types.int32,
                   types.Array(types.int32, 1, 'C'),
                   types.complex64]
        self._test_as_arguments(fe_args)

    def test_two_arrays(self):
        fe_args = [types.Array(types.int32, 1, 'C')] * 2
        self._test_as_arguments(fe_args)

    def test_two_0d_arrays(self):
        fe_args = [types.Array(types.int32, 0, 'C')] * 2
        self._test_as_arguments(fe_args)

    def test_tuples(self):
        fe_args = [types.UniTuple(types.int32, 2),
                   types.UniTuple(types.int32, 3)]
        self._test_as_arguments(fe_args)
        # Tuple of struct-likes
        arrty = types.Array(types.int32, 1, 'C')
        fe_args = [types.UniTuple(arrty, 2),
                   types.UniTuple(arrty, 3)]
        self._test_as_arguments(fe_args)
        # Nested tuple
        fe_args = [types.UniTuple(types.UniTuple(types.int32, 2), 3)]
        self._test_as_arguments(fe_args)

    def test_empty_tuples(self):
        # Empty tuple
        fe_args = [types.UniTuple(types.int16, 0),
                   types.Tuple(()),
                   types.int32]
        self._test_as_arguments(fe_args)

    def test_nested_empty_tuples(self):
        fe_args = [types.int32,
                   types.UniTuple(types.Tuple(()), 2),
                   types.int64]
        self._test_as_arguments(fe_args)


if __name__ == '__main__':
    unittest.main()
