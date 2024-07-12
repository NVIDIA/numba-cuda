# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from llvmlite import ir
from numba.core import cgutils, config
from numba.core.runtime.nrtdynmod import (
    _define_atomic_cas,
    _define_atomic_inc_dec,
    _define_nrt_incref,
    _define_nrt_meminfo_data,
    _pointer_type,
    incref_decref_ty,
)




def _define_nrt_decref(module, atomic_decr):
    """
    Implement NRT_decref in the module
    """
    fn_decref = cgutils.get_or_insert_function(
        module, incref_decref_ty, "NRT_decref"
    )
    # Cannot inline this for refcount pruning to work

    fn_decref.attributes.add("noinline")
    calldtor = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), [_pointer_type, _pointer_type]),
        name="NRT_MemInfo_call_dtor",
    )

    builder = ir.IRBuilder(fn_decref.append_basic_block())
    [ptr] = fn_decref.args
    is_null = builder.icmp_unsigned(
        "==", ptr, cgutils.get_null_value(ptr.type)
    )
    with cgutils.if_unlikely(builder, is_null):
        builder.ret_void()

    # For memory fence usage, see https://llvm.org/docs/Atomics.html

    # A release fence is used before the relevant write operation.
    # No-op on x86.  On POWER, it lowers to lwsync.
    #    builder.fence("release")

    word_ptr = builder.bitcast(ptr, atomic_decr.args[0].type)

    if config.DEBUG_NRT:
        cgutils.printf(
            builder, "*** NRT_Decref %zu [%p]\n", builder.load(word_ptr), ptr
        )
    newrefct = builder.call(
        atomic_decr, [word_ptr]  # nvvmutils.atomic_dec_int64, etc
    )

    refct_eq_0 = builder.icmp_unsigned(
        "==", newrefct, ir.Constant(newrefct.type, 0)
    )
    with cgutils.if_unlikely(builder, refct_eq_0):
        # An acquire fence is used after the relevant read operation.
        # No-op on x86.  On POWER, it lowers to lwsync.
        #        builder.fence("acquire")
        builder.call(
            calldtor,
            [
                ptr,
                ir.Constant.inttoptr(
                    ir.Constant(ir.IntType(64), memsys), _pointer_type
                ),
            ],
        )
    builder.ret_void()


def create_nrt_module(ctx):
    """
    Create an IR module defining the LLVM NRT functions.
    A (IR module, library) tuple is returned.
    """
    codegen = ctx.codegen()
    library = codegen.create_library("nrt")

    # Implement LLVM module with atomic ops
    ir_mod = library.create_ir_module("nrt_module")

    atomic_inc = _define_atomic_inc_dec(ir_mod, "add", ordering="monotonic")
    atomic_dec = _define_atomic_inc_dec(ir_mod, "sub", ordering="monotonic")
    _define_atomic_cas(ir_mod, ordering="monotonic")

    _define_nrt_meminfo_data(ir_mod)
    _define_nrt_incref(ir_mod, atomic_inc)
    _define_nrt_decref(ir_mod, atomic_dec)

    #    _define_nrt_unresolved_abort(ctx, ir_mod)
    return ir_mod, library


def compile_nrt_functions(ctx):
    """
    Compile all LLVM NRT functions and return a library containing them.
    The library is created using the given target context.
    """
    ir_mod, library = create_nrt_module(ctx)

    library.add_ir_module(ir_mod)
    library.finalize()
    breakpoint()

    return library


# compile NRT functions written in lowering into a PTX string
#library = compile_nrt_functions(descriptor.cuda_target.target_context)
#_cuda_nrt_library = compile_nrt_functions(
#    descriptor.cuda_target.target_context
#)
#numba_cuda_runtime = _cuda_nrt_library._get_ptxes()[0].encode()

def NRT_MemSys_new():
    # TODO
    pass


memsys = NRT_MemSys_new()
numba_cuda_runtime=None
