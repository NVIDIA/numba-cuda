from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.extending import intrinsic, overload
from numba.core.errors import NumbaTypeError
from numba.cuda.api_util import normalize_indices


def ldca(array, i):
    """Generate a `ld.global.ca` instruction for element `i` of an array."""


def ldcg(array, i):
    """Generate a `ld.global.cg` instruction for element `i` of an array."""


def ldcs(array, i):
    """Generate a `ld.global.cs` instruction for element `i` of an array."""


def ldlu(array, i):
    """Generate a `ld.global.lu` instruction for element `i` of an array."""


def ldcv(array, i):
    """Generate a `ld.global.cv` instruction for element `i` of an array."""


def stcg(array, i, value):
    """Generate a `st.global.cg` instruction for element `i` of an array."""


def stcs(array, i, value):
    """Generate a `st.global.cs` instruction for element `i` of an array."""


def stwb(array, i, value):
    """Generate a `st.global.wb` instruction for element `i` of an array."""


def stwt(array, i, value):
    """Generate a `st.global.wt` instruction for element `i` of an array."""


def ld_cache_operator(operator):
    @intrinsic
    def impl(typingctx, array, index):
        if not isinstance(array, types.Array):
            msg = f"ldcs operates on arrays. Got type {array}"
            raise NumbaTypeError(msg)

        # Need to validate bitwidth

        # Need to validate indices

        signature = array.dtype(array, index)

        def codegen(context, builder, sig, args):
            array_type, index_type = sig.args
            loaded_type = context.get_value_type(array_type.dtype)
            ptr_type = loaded_type.as_pointer()
            ldcs_type = ir.FunctionType(loaded_type, [ptr_type])

            array, indices = args

            index_type, indices = normalize_indices(context, builder,
                                                    index_type, indices,
                                                    array_type,
                                                    array_type.dtype)
            array_struct = context.make_array(array_type)(context, builder,
                                                          value=array)
            ptr = cgutils.get_item_pointer(context, builder, array_type,
                                           array_struct, indices,
                                           wraparound=True)

            bitwidth = array_type.dtype.bitwidth
            inst = f"ld.global.{operator}.b{bitwidth}"
            # See
            # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#restricted-use-of-sub-word-sizes
            # for background on the choice of "r" for 8-bit operands - there is
            # no constraint for 8-bit operands, but the operand for loads and
            # stores is permitted to be greater than 8 bits.
            constraint_map = {
                1: "b",
                8: "r",
                16: "h",
                32: "r",
                64: "l",
                128: "q"
            }
            constraints = f"={constraint_map[bitwidth]},l"
            ldcs = ir.InlineAsm(ldcs_type, f"{inst} $0, [$1];", constraints)
            return builder.call(ldcs, [ptr])

        return signature, codegen

    return impl


ldca_intrinsic = ld_cache_operator("ca")
ldcg_intrinsic = ld_cache_operator("cg")
ldcs_intrinsic = ld_cache_operator("cs")
ldlu_intrinsic = ld_cache_operator("lu")
ldcv_intrinsic = ld_cache_operator("cv")


def st_cache_operator(operator):
    @intrinsic
    def impl(typingctx, array, index, value):
        if not isinstance(array, types.Array):
            msg = f"ldcs operates on arrays. Got type {array}"
            raise NumbaTypeError(msg)

        # Need to validate bitwidth

        # Need to validate indices

        signature = types.void(array, index, value)

        def codegen(context, builder, sig, args):
            array_type, index_type, value_type = sig.args
            stored_type = context.get_value_type(array_type.dtype)
            ptr_type = stored_type.as_pointer()
            stcs_type = ir.FunctionType(ir.VoidType(), [ptr_type, stored_type])

            array, indices, value = args

            index_type, indices = normalize_indices(context, builder,
                                                    index_type, indices,
                                                    array_type,
                                                    array_type.dtype)
            array_struct = context.make_array(array_type)(context, builder,
                                                          value=array)
            ptr = cgutils.get_item_pointer(context, builder, array_type,
                                           array_struct, indices,
                                           wraparound=True)

            casted_value = context.cast(builder, value, value_type,
                                        array_type.dtype)

            bitwidth = array_type.dtype.bitwidth
            inst = f"st.global.{operator}.b{bitwidth}"
            constraint_map = {
                1: "b",
                8: "r",
                16: "h",
                32: "r",
                64: "l",
                128: "q"
            }
            constraints = f"l,{constraint_map[bitwidth]},~{{memory}}"
            stcs = ir.InlineAsm(stcs_type, f"{inst} [$0], $1;", constraints)
            builder.call(stcs, [ptr, casted_value])

        return signature, codegen

    return impl


stcg_intrinsic = st_cache_operator("cg")
stcs_intrinsic = st_cache_operator("cs")
stwb_intrinsic = st_cache_operator("wb")
stwt_intrinsic = st_cache_operator("wt")


@overload(ldca, target='cuda')
def ol_ldca(array, i):
    def impl(array, i):
        return ldca_intrinsic(array, i)
    return impl


@overload(ldcg, target='cuda')
def ol_ldcg(array, i):
    def impl(array, i):
        return ldcg_intrinsic(array, i)
    return impl


@overload(ldcs, target='cuda')
def ol_ldcs(array, i):
    def impl(array, i):
        return ldcs_intrinsic(array, i)
    return impl


@overload(ldlu, target='cuda')
def ol_ldlu(array, i):
    def impl(array, i):
        return ldlu_intrinsic(array, i)
    return impl


@overload(ldcv, target='cuda')
def ol_ldcv(array, i):
    def impl(array, i):
        return ldcv_intrinsic(array, i)
    return impl


@overload(stcg, target='cuda')
def ol_stcg(array, i, value):
    def impl(array, i, value):
        return stcg_intrinsic(array, i, value)
    return impl


@overload(stcs, target='cuda')
def ol_stcs(array, i, value):
    def impl(array, i, value):
        return stcs_intrinsic(array, i, value)
    return impl


@overload(stwb, target='cuda')
def ol_stwb(array, i, value):
    def impl(array, i, value):
        return stwb_intrinsic(array, i, value)
    return impl


@overload(stwt, target='cuda')
def ol_stwt(array, i, value):
    def impl(array, i, value):
        return stwt_intrinsic(array, i, value)
    return impl
