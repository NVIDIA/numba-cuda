# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from llvmlite import ir

from numba import cuda, types
from numba.cuda import cgutils
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cuda.typing import signature
from numba.core.extending import overload_attribute, overload_method
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic


# -------------------------------------------------------------------------------
# Grid functions


def _type_grid_function(ndim):
    val = ndim.literal_value
    if val == 1:
        restype = types.int64
    elif val in (2, 3):
        restype = types.UniTuple(types.int64, val)
    else:
        raise ValueError("argument can only be 1, 2, 3")

    return signature(restype, types.int32)


@intrinsic
def grid(typingctx, ndim):
    """grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """

    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        if restype == types.int64:
            return nvvmutils.get_global_id(builder, dim=1)
        elif isinstance(restype, types.UniTuple):
            ids = nvvmutils.get_global_id(builder, dim=restype.count)
            return cgutils.pack_array(builder, ids)

    return sig, codegen


@intrinsic
def gridsize(typingctx, ndim):
    """gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """

    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    def _nthreads_for_dim(builder, dim):
        i64 = ir.IntType(64)
        ntid = nvvmutils.call_sreg(builder, f"ntid.{dim}")
        nctaid = nvvmutils.call_sreg(builder, f"nctaid.{dim}")
        return builder.mul(builder.sext(ntid, i64), builder.sext(nctaid, i64))

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        nx = _nthreads_for_dim(builder, "x")

        if restype == types.int64:
            return nx
        elif isinstance(restype, types.UniTuple):
            ny = _nthreads_for_dim(builder, "y")

            if restype.count == 2:
                return cgutils.pack_array(builder, (nx, ny))
            elif restype.count == 3:
                nz = _nthreads_for_dim(builder, "z")
                return cgutils.pack_array(builder, (nx, ny, nz))

    return sig, codegen


@intrinsic
def _warpsize(typingctx):
    sig = signature(types.int32)

    def codegen(context, builder, sig, args):
        return nvvmutils.call_sreg(builder, "warpsize")

    return sig, codegen


@overload_attribute(types.Module(cuda), "warpsize", target="cuda")
def cuda_warpsize(mod):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """

    def get(mod):
        return _warpsize()

    return get


# -------------------------------------------------------------------------------
# syncthreads


@intrinsic
def syncthreads(typingctx):
    """
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    """
    sig = signature(types.none)

    def codegen(context, builder, sig, args):
        fname = "llvm.nvvm.barrier0"
        lmod = builder.module
        fnty = ir.FunctionType(ir.VoidType(), ())
        sync = cgutils.get_or_insert_function(lmod, fnty, fname)
        builder.call(sync, ())
        return context.get_dummy_value()

    return sig, codegen


def _syncthreads_predicate(typingctx, predicate, fname):
    if not isinstance(predicate, types.Integer):
        return None

    sig = signature(types.i4, types.i4)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ir.IntType(32), (ir.IntType(32),))
        sync = cgutils.get_or_insert_function(builder.module, fnty, fname)
        return builder.call(sync, args)

    return sig, codegen


@intrinsic
def syncthreads_count(typingctx, predicate):
    """
    syncthreads_count(predicate)

    An extension to numba.cuda.syncthreads where the return value is a count
    of the threads where predicate is true.
    """
    fname = "llvm.nvvm.barrier0.popc"
    return _syncthreads_predicate(typingctx, predicate, fname)


@intrinsic
def syncthreads_and(typingctx, predicate):
    """
    syncthreads_and(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for all threads or 0 otherwise.
    """
    fname = "llvm.nvvm.barrier0.and"
    return _syncthreads_predicate(typingctx, predicate, fname)


@intrinsic
def syncthreads_or(typingctx, predicate):
    """
    syncthreads_or(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for any thread or 0 otherwise.
    """
    fname = "llvm.nvvm.barrier0.or"
    return _syncthreads_predicate(typingctx, predicate, fname)


@overload_method(types.Integer, "bit_count", target="cuda")
def integer_bit_count(i):
    return lambda i: cuda.popc(i)


# -------------------------------------------------------------------------------
# Warp shuffle functions
#
# References:
#
# - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
# - https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-movement
#
# Notes:
#
# - The public CUDA C/C++ and Numba Python APIs for these intrinsics use
#   different names for parameters to the NVVM IR specification. So that we
#   can correlate the implementation with the documentation, the @intrinsic
#   API functions map the public API arguments to the NVVM intrinsic
#   arguments.
# - The NVVM IR specification requires some of the parameters (e.g. mode) to be
#   constants. It's therefore essential that we pass in some values to the
#   shfl_sync_intrinsic function (e.g. the mode and c values).
# - Normally parameters for intrinsic functions in Numba would be given the
#   same name as used in the API, and would contain a type. However, because we
#   have to pass in some values and some times (and there is divergence between
#   the names in the intrinsic documentation and the public APIs) we instead
#   follow the convention of naming shfl_sync_intrinsic parameters with a
#   suffix of _type or _value depending on whether they contain a type or a
#   value.


@intrinsic
def shfl_sync(typingctx, mask, value, src_lane):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``src_lane``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 0
    a_type = value
    b_type = src_lane
    c_value = 0x1F
    return shfl_sync_intrinsic(
        typingctx, membermask_type, mode_value, a_type, b_type, c_value
    )


@intrinsic
def shfl_up_sync(typingctx, mask, value, delta):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid - delta)``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 1
    a_type = value
    b_type = delta
    c_value = 0
    return shfl_sync_intrinsic(
        typingctx, membermask_type, mode_value, a_type, b_type, c_value
    )


@intrinsic
def shfl_down_sync(typingctx, mask, value, delta):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid + delta)``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 2
    a_type = value
    b_type = delta
    c_value = 0x1F
    return shfl_sync_intrinsic(
        typingctx, membermask_type, mode_value, a_type, b_type, c_value
    )


@intrinsic
def shfl_xor_sync(typingctx, mask, value, lane_mask):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid ^ lane_mask)``.
    """
    membermask_type = mask
    mode_value = 3
    a_type = value
    b_type = lane_mask
    c_value = 0x1F
    return shfl_sync_intrinsic(
        typingctx, membermask_type, mode_value, a_type, b_type, c_value
    )


def shfl_sync_intrinsic(
    typingctx,
    membermask_type,
    mode_value,
    a_type,
    b_type,
    c_value,
):
    if a_type not in (types.i4, types.i8, types.f4, types.f8):
        raise TypingError(
            "shfl_sync only supports 32- and 64-bit ints and floats"
        )

    def codegen(context, builder, sig, args):
        """
        The NVVM shfl_sync intrinsic only supports i32, but the CUDA C/C++
        intrinsic supports both 32- and 64-bit ints and floats, so for feature
        parity, i32, i64, f32, and f64 are implemented. Floats by way of
        bitcasting the float to an int, then shuffling, then bitcasting
        back."""
        membermask, a, b = args

        # Types
        a_type = sig.args[1]
        return_type = context.get_value_type(sig.return_type)
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)

        if a_type in types.real_domain:
            a = builder.bitcast(a, ir.IntType(a_type.bitwidth))

        # NVVM intrinsic definition
        arg_types = (i32, i32, i32, i32, i32)
        shfl_return_type = ir.LiteralStructType((i32, ir.IntType(1)))
        fnty = ir.FunctionType(shfl_return_type, arg_types)

        fname = "llvm.nvvm.shfl.sync.i32"
        shfl_sync = cgutils.get_or_insert_function(builder.module, fnty, fname)

        # Intrinsic arguments
        mode = ir.Constant(i32, mode_value)
        c = ir.Constant(i32, c_value)
        membermask = builder.trunc(membermask, i32)
        b = builder.trunc(b, i32)

        if a_type.bitwidth == 32:
            a = builder.trunc(a, i32)
            ret = builder.call(shfl_sync, (membermask, mode, a, b, c))
            d = builder.extract_value(ret, 0)
        else:
            # Handle 64-bit values by shuffling as two 32-bit values and
            # packing the result into 64 bits.

            # Extract high and low parts
            lo = builder.trunc(a, i32)
            a_lshr = builder.lshr(a, ir.Constant(i64, 32))
            hi = builder.trunc(a_lshr, i32)

            # Shuffle individual parts
            ret_lo = builder.call(shfl_sync, (membermask, mode, lo, b, c))
            ret_hi = builder.call(shfl_sync, (membermask, mode, hi, b, c))

            # Combine individual result parts into a 64-bit result
            d_lo = builder.extract_value(ret_lo, 0)
            d_hi = builder.extract_value(ret_hi, 0)
            d_lo_64 = builder.zext(d_lo, i64)
            d_hi_64 = builder.zext(d_hi, i64)
            d_shl = builder.shl(d_hi_64, ir.Constant(i64, 32))
            d = builder.or_(d_shl, d_lo_64)

        return builder.bitcast(d, return_type)

    sig = signature(a_type, membermask_type, a_type, b_type)

    return sig, codegen
