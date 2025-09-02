# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from llvmlite import ir
from collections import namedtuple
from warnings import warn, catch_warnings, simplefilter
import copy

from numba.core import ir as numba_ir
from numba.core import (
    types,
    funcdesc,
    bytecode,
    cpu,
)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.errors import NumbaWarning, NumbaInvalidConfigWarning
from numba.cuda.core.interpreter import Interpreter
from numba.cuda.core import inline_closurecall

from numba.cuda import cgutils, typing, lowering, nvvmutils, utils
from numba.cuda.api import get_current_device
from numba.cuda.codegen import ExternalCodeLibrary
from numba.cuda.core import sigutils, postproc, config
from numba.cuda.cudadrv import nvvm, nvrtc
from numba.cuda.descriptor import cuda_target
from numba.cuda.flags import CUDAFlags
from numba.cuda.target import CUDACABICallConv
from numba.cuda.core.compiler import CompilerBase
from numba.cuda.core.compiler_machinery import (
    FunctionPass,
    LoweringPass,
    PassManager,
    register_pass,
)
from numba.cuda.core.untyped_passes import (
    TranslateByteCode,
    FixupArgs,
    IRProcessing,
    DeadBranchPrune,
    RewriteSemanticConstants,
    InlineClosureLikes,
    GenericRewrites,
    WithLifting,
    InlineInlinables,
    FindLiterallyCalls,
    MakeFunctionToJitFunction,
    LiteralUnroll,
    ReconstructSSA,
    RewriteDynamicRaises,
    LiteralPropagationSubPipelinePass,
)
from numba.cuda.core.typed_passes import (
    BaseNativeLowering,
    NativeLowering,
    AnnotateTypes,
    IRLegalization,
    NopythonTypeInference,
    NopythonRewrites,
    InlineOverloads,
    PreLowerStripPhis,
    NoPythonSupportedFeatureValidation,
)


_LowerResult = namedtuple(
    "_LowerResult",
    [
        "fndesc",
        "call_helper",
        "cfunc",
        "env",
    ],
)


def sanitize_compile_result_entries(entries):
    keys = set(entries.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*badnames)
    missing = fieldset - keys
    for k in missing:
        entries[k] = None
    # Avoid keeping alive traceback variables
    err = entries["typing_error"]
    if err is not None:
        entries["typing_error"] = err.with_traceback(None)
    return entries


def run_frontend(func, inline_closures=False, emit_dels=False):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.

    If inline_closures is Truthy then closure inlining will be run
    If emit_dels is Truthy the ir.Del nodes will be emitted appropriately
    """
    # XXX make this a dedicated Pipeline?
    func_id = bytecode.FunctionIdentity.from_function(func)
    interp = Interpreter(func_id)
    bc = bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    if inline_closures:
        inline_pass = inline_closurecall.InlineClosureCallPass(
            func_ir, cpu.ParallelOptions(False), {}, False
        )
        inline_pass.run()
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run(emit_dels)
    return func_ir


class DefaultPassBuilder(object):
    """
    This is the default pass builder, it contains the "classic" default
    pipelines as pre-canned PassManager instances:
      - nopython
      - objectmode
      - interpreted
      - typed
      - untyped
      - nopython lowering
    """

    @staticmethod
    def define_nopython_pipeline(state, name="nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        # compose pipeline from untyped, typed and lowering parts
        dpb = DefaultPassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name="nopython_lowering"):
        pm = PassManager(name)
        # legalise
        pm.add_pass(
            NoPythonSupportedFeatureValidation,
            "ensure features that are in use are in a valid form",
        )
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        # Annotate only once legalized
        pm.add_pass(AnnotateTypes, "annotate types")
        # lower
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(CUDABackend, "nopython mode backend")
        pm.finalize()
        return pm

    @staticmethod
    def define_parfor_gufunc_nopython_lowering_pipeline(
        state, name="parfor_gufunc_nopython_lowering"
    ):
        pm = PassManager(name)
        # legalise
        pm.add_pass(
            NoPythonSupportedFeatureValidation,
            "ensure features that are in use are in a valid form",
        )
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        # Annotate only once legalized
        pm.add_pass(AnnotateTypes, "annotate types")
        # lower
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(CUDABackend, "nopython mode backend")
        pm.finalize()
        return pm

    @staticmethod
    def define_typed_pipeline(state, name="typed"):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")

        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")

        pm.finalize()
        return pm

    @staticmethod
    def define_untyped_pipeline(state, name="untyped"):
        """Returns an untyped part of the nopython pipeline"""
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # inline closures early in case they are using nonlocal's
        # see issue #6585.
        pm.add_pass(
            InlineClosureLikes, "inline calls to locally defined closures"
        )

        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
            pm.add_pass(GenericRewrites, "nopython rewrites")

        pm.add_pass(RewriteDynamicRaises, "rewrite dynamic raises")

        # convert any remaining closures into functions
        pm.add_pass(
            MakeFunctionToJitFunction,
            "convert make_function into JIT functions",
        )
        # inline functions that have been determined as inlinable and rerun
        # branch pruning, this needs to be run after closures are inlined as
        # the IR repr of a closure masks call sites if an inlinable is called
        # inside a closure
        pm.add_pass(InlineInlinables, "inline inlinable functions")
        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(FindLiterallyCalls, "find literally calls")
        pm.add_pass(LiteralUnroll, "handles literal_unroll")

        if state.flags.enable_ssa:
            pm.add_pass(ReconstructSSA, "ssa")

        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(LiteralPropagationSubPipelinePass, "Literal propagation")

        pm.finalize()
        return pm


# The CUDACompileResult (CCR) has a specially-defined entry point equal to its
# id.  This is because the entry point is used as a key into a dict of
# overloads by the base dispatcher. The id of the CCR is the only small and
# unique property of a CUDACompileResult in the CUDA target (cf. the CPU target,
# which uses its entry_point, which is a pointer value).
#
# This does feel a little hackish, and there are two ways in which this could
# be improved:
#
# 1. We could change the CUDACompileResult so that each instance has its own
#    unique ID that can be used as a key - e.g. a count, similar to the way in
#    which types have unique counts.
# 2. At some future time when kernel launch uses a compiled function, the entry
#    point will no longer need to be a synthetic value, but will instead be a
#    pointer to the compiled function as in the CPU target.

CR_FIELDS = [
    "typing_context",
    "target_context",
    "entry_point",
    "typing_error",
    "type_annotation",
    "signature",
    "objectmode",
    "lifted",
    "fndesc",
    "library",
    "call_helper",
    "environment",
    "metadata",
    # List of functions to call to initialize on unserialization
    # (i.e cache load).
    "reload_init",
    "referenced_envs",
]


class CUDACompileResult(namedtuple("_CompileResult", CR_FIELDS)):
    """
    A structure holding results from the compilation of a function.
    """

    __slots__ = ()

    @property
    def entry_point(self):
        return id(self)

    def _reduce(self):
        """
        Reduce a CompileResult to picklable components.
        """
        libdata = self.library.serialize_using_object_code()
        # Make it (un)picklable efficiently
        typeann = str(self.type_annotation)
        fndesc = self.fndesc
        # Those don't need to be pickled and may fail
        fndesc.typemap = fndesc.calltypes = None
        # The CUDA target does not reference environments
        referenced_envs = tuple()
        return (
            libdata,
            self.fndesc,
            self.environment,
            self.signature,
            self.objectmode,
            self.lifted,
            typeann,
            self.reload_init,
            referenced_envs,
        )

    @classmethod
    def _rebuild(
        cls,
        target_context,
        libdata,
        fndesc,
        env,
        signature,
        objectmode,
        lifted,
        typeann,
        reload_init,
        referenced_envs,
    ):
        if reload_init:
            # Re-run all
            for fn in reload_init:
                fn()

        library = target_context.codegen().unserialize_library(libdata)
        cfunc = target_context.get_executable(library, fndesc, env)
        cr = cls(
            target_context=target_context,
            typing_context=target_context.typing_context,
            library=library,
            environment=env,
            entry_point=cfunc,
            fndesc=fndesc,
            type_annotation=typeann,
            signature=signature,
            objectmode=objectmode,
            lifted=lifted,
            typing_error=None,
            call_helper=None,
            metadata=None,  # Do not store, arbitrary & potentially large!
            reload_init=reload_init,
            referenced_envs=referenced_envs,
        )

        # Load Environments
        for env in referenced_envs:
            library.codegen.set_env(env.env_name, env)

        return cr

    @property
    def codegen(self):
        return self.target_context.codegen()

    def dump(self, tab=""):
        print(f"{tab}DUMP {type(self).__name__} {self.entry_point}")
        self.signature.dump(tab=tab + "  ")
        print(f"{tab}END DUMP")


def cuda_compile_result(**entries):
    entries = sanitize_compile_result_entries(entries)
    return CUDACompileResult(**entries)


@register_pass(mutates_CFG=True, analysis_only=False)
class CUDABackend(LoweringPass):
    _name = "cuda_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        lowered = state["cr"]
        signature = typing.signature(state.return_type, *state.args)

        state.cr = cuda_compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            fndesc=lowered.fndesc,
        )
        return True


@register_pass(mutates_CFG=False, analysis_only=False)
class CreateLibrary(LoweringPass):
    """
    Create a CUDACodeLibrary for the NativeLowering pass to populate. The
    NativeLowering pass will create a code library if none exists, but we need
    to set it up with nvvm_options from the flags if they are present.
    """

    _name = "create_library"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        codegen = state.targetctx.codegen()
        name = state.func_id.func_qualname
        nvvm_options = state.flags.nvvm_options
        max_registers = state.flags.max_registers
        lto = state.flags.lto
        state.library = codegen.create_library(
            name,
            nvvm_options=nvvm_options,
            max_registers=max_registers,
            lto=lto,
        )
        # Enable object caching upfront so that the library can be serialized.
        state.library.enable_object_caching()

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class CUDANativeLowering(BaseNativeLowering):
    """Lowering pass for a CUDA native function IR described solely in terms of
    Numba's standard `numba.core.ir` nodes."""

    _name = "cuda_native_lowering"

    @property
    def lowering_class(self):
        return lowering.CUDALower


class CUDABytecodeInterpreter(Interpreter):
    # Based on the superclass implementation, but names the resulting variable
    # "$bool<N>" instead of "bool<N>" - see Numba PR #9888:
    # https://github.com/numba/numba/pull/9888
    #
    # This can be removed once that PR is available in an upstream Numba
    # release.
    def _op_JUMP_IF(self, inst, pred, iftrue):
        brs = {
            True: inst.get_jump_target(),
            False: inst.next,
        }
        truebr = brs[iftrue]
        falsebr = brs[not iftrue]

        name = "$bool%s" % (inst.offset)
        gv_fn = numba_ir.Global("bool", bool, loc=self.loc)
        self.store(value=gv_fn, name=name)

        callres = numba_ir.Expr.call(
            self.get(name), (self.get(pred),), (), loc=self.loc
        )

        pname = "$%spred" % (inst.offset)
        predicate = self.store(value=callres, name=pname)
        bra = numba_ir.Branch(
            cond=predicate, truebr=truebr, falsebr=falsebr, loc=self.loc
        )
        self.current_block.append(bra)


@register_pass(mutates_CFG=True, analysis_only=False)
class CUDATranslateBytecode(FunctionPass):
    _name = "cuda_translate_bytecode"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_id = state["func_id"]
        bc = state["bc"]
        interp = CUDABytecodeInterpreter(func_id)
        func_ir = interp.interpret(bc)
        state["func_ir"] = func_ir
        return True


class CUDACompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager("cuda")

        untyped_passes = dpb.define_untyped_pipeline(self.state)

        # Rather than replicating the whole untyped passes definition in
        # numba-cuda, it seems cleaner to take the pass list and replace the
        # TranslateBytecode pass with our own.

        def replace_translate_pass(implementation, description):
            if implementation is TranslateByteCode:
                return (CUDATranslateBytecode, description)
            else:
                return (implementation, description)

        cuda_untyped_passes = [
            replace_translate_pass(implementation, description)
            for implementation, description in untyped_passes.passes
        ]

        pm.passes.extend(cuda_untyped_passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_cuda_lowering_pipeline(self, state):
        pm = PassManager("cuda_lowering")
        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")

        # lower
        pm.add_pass(CreateLibrary, "create library")
        pm.add_pass(CUDANativeLowering, "cuda native lowering")
        pm.add_pass(CUDABackend, "cuda backend")

        pm.finalize()
        return pm


def compile_extra(
    typingctx,
    targetctx,
    func,
    args,
    return_type,
    flags,
    locals,
    library=None,
    pipeline_class=CUDACompiler,
):
    """Compiler entry point

    Parameter
    ---------
    typingctx :
        typing context
    targetctx :
        target context
    func : function
        the python function to be compiled
    args : tuple, list
        argument types
    return_type :
        Use ``None`` to indicate void return
    flags : numba.compiler.Flags
        compiler flags
    library : numba.codegen.CodeLibrary
        Used to store the compiled code.
        If it is ``None``, a new CodeLibrary is used.
    pipeline_class : type like numba.compiler.CompilerBase
        compiler pipeline
    """
    pipeline = pipeline_class(
        typingctx, targetctx, library, args, return_type, flags, locals
    )
    return pipeline.compile_extra(func)


def compile_ir(
    typingctx,
    targetctx,
    func_ir,
    args,
    return_type,
    flags,
    locals,
    lifted=(),
    lifted_from=None,
    is_lifted_loop=False,
    library=None,
    pipeline_class=CUDACompiler,
):
    """
    Compile a function with the given IR.

    For internal use only.
    """

    # This is a special branch that should only run on IR from a lifted loop
    if is_lifted_loop:
        # This code is pessimistic and costly, but it is a not often trodden
        # path and it will go away once IR is made immutable. The problem is
        # that the rewrite passes can mutate the IR into a state that makes
        # it possible for invalid tokens to be transmitted to lowering which
        # then trickle through into LLVM IR and causes RuntimeErrors as LLVM
        # cannot compile it. As a result the following approach is taken:
        # 1. Create some new flags that copy the original ones but switch
        #    off rewrites.
        # 2. Compile with 1. to get a compile result
        # 3. Try and compile another compile result but this time with the
        #    original flags (and IR being rewritten).
        # 4. If 3 was successful, use the result, else use 2.

        # create flags with no rewrites
        norw_flags = copy.deepcopy(flags)
        norw_flags.no_rewrites = True

        def compile_local(the_ir, the_flags):
            pipeline = pipeline_class(
                typingctx,
                targetctx,
                library,
                args,
                return_type,
                the_flags,
                locals,
            )
            return pipeline.compile_ir(
                func_ir=the_ir, lifted=lifted, lifted_from=lifted_from
            )

        # compile with rewrites off, IR shouldn't be mutated irreparably
        norw_cres = compile_local(func_ir.copy(), norw_flags)

        # try and compile with rewrites on if no_rewrites was not set in the
        # original flags, IR might get broken but we've got a CompileResult
        # that's usable from above.
        rw_cres = None
        if not flags.no_rewrites:
            # Suppress warnings in compilation retry
            with catch_warnings():
                simplefilter("ignore", NumbaWarning)
                try:
                    rw_cres = compile_local(func_ir.copy(), flags)
                except Exception:
                    pass
        # if the rewrite variant of compilation worked, use it, else use
        # the norewrites backup
        if rw_cres is not None:
            cres = rw_cres
        else:
            cres = norw_cres
        return cres

    else:
        pipeline = pipeline_class(
            typingctx, targetctx, library, args, return_type, flags, locals
        )
        return pipeline.compile_ir(
            func_ir=func_ir, lifted=lifted, lifted_from=lifted_from
        )


def compile_internal(
    typingctx, targetctx, library, func, args, return_type, flags, locals
):
    """
    For internal use only.
    """
    pipeline = CUDACompiler(
        typingctx, targetctx, library, args, return_type, flags, locals
    )
    return pipeline.compile_extra(func)


@global_compiler_lock
def compile_cuda(
    pyfunc,
    return_type,
    args,
    debug=False,
    lineinfo=False,
    forceinline=False,
    fastmath=False,
    nvvm_options=None,
    cc=None,
    max_registers=None,
    lto=False,
):
    if cc is None:
        raise ValueError("Compute Capability must be supplied")

    from .descriptor import cuda_target

    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context

    flags = CUDAFlags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True

    # Both debug and lineinfo turn on debug information in the compiled code,
    # but we keep them separate arguments in case we later want to overload
    # some other behavior on the debug flag. In particular, -opt=3 is not
    # supported with debug enabled, and enabling only lineinfo should not
    # affect the error model.
    if debug or lineinfo:
        flags.debuginfo = True

    if lineinfo:
        flags.dbg_directives_only = True

    if debug:
        flags.error_model = "python"
        flags.dbg_extend_lifetimes = True
    else:
        flags.error_model = "numpy"

    if forceinline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if nvvm_options:
        flags.nvvm_options = nvvm_options
    flags.compute_capability = cc
    flags.max_registers = max_registers
    flags.lto = lto

    # Run compilation pipeline
    from numba.core.target_extension import target_override

    with target_override("cuda"):
        cres = compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=CUDACompiler,
        )

    library = cres.library
    library.finalize()

    return cres


def cabi_wrap_function(
    context, lib, fndesc, wrapper_function_name, nvvm_options
):
    """
    Wrap a Numba ABI function in a C ABI wrapper at the NVVM IR level.

    The C ABI wrapper will have the same name as the source Python function.
    """
    # The wrapper will be contained in a new library that links to the wrapped
    # function's library
    library = lib.codegen.create_library(
        f"{lib.name}_function_",
        entry_name=wrapper_function_name,
        nvvm_options=nvvm_options,
    )
    library.add_linking_library(lib)

    # Determine the caller (C ABI) and wrapper (Numba ABI) function types
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    c_call_conv = CUDACABICallConv(context)
    wrapfnty = c_call_conv.get_function_type(restype, argtypes)
    fnty = context.call_conv.get_function_type(fndesc.restype, argtypes)

    # Create a new module and declare the callee
    wrapper_module = context.create_module("cuda.cabi.wrapper")
    func = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)

    # Define the caller - populate it with a call to the callee and return
    # its return value

    wrapfn = ir.Function(wrapper_module, wrapfnty, wrapper_function_name)
    builder = ir.IRBuilder(wrapfn.append_basic_block(""))

    arginfo = context.get_arg_packer(argtypes)
    callargs = arginfo.from_arguments(builder, wrapfn.args)
    # We get (status, return_value), but we ignore the status since we
    # can't propagate it through the C ABI anyway
    _, return_value = context.call_conv.call_function(
        builder, func, restype, argtypes, callargs
    )
    builder.ret(return_value)

    if config.DUMP_LLVM:
        utils.dump_llvm(fndesc, wrapper_module)

    library.add_ir_module(wrapper_module)
    library.finalize()
    return library


def kernel_fixup(kernel, debug):
    if debug:
        exc_helper = add_exception_store_helper(kernel)

    # Pass 1 - replace:
    #
    #    ret <value>
    #
    # with:
    #
    #    exc_helper(<value>)
    #    ret void

    for block in kernel.blocks:
        for i, inst in enumerate(block.instructions):
            if isinstance(inst, ir.Ret):
                old_ret = block.instructions.pop()
                block.terminator = None

                # The original return's metadata will be set on the new
                # instructions in order to preserve debug info
                metadata = old_ret.metadata

                builder = ir.IRBuilder(block)
                if debug:
                    status_code = old_ret.operands[0]
                    exc_helper_call = builder.call(exc_helper, (status_code,))
                    exc_helper_call.metadata = metadata

                new_ret = builder.ret_void()
                new_ret.metadata = old_ret.metadata

                # Need to break out so we don't carry on modifying what we are
                # iterating over. There can only be one return in a block
                # anyway.
                break

    # Pass 2: remove stores of null pointer to return value argument pointer

    return_value = kernel.args[0]

    for block in kernel.blocks:
        remove_list = []

        # Find all stores first
        for inst in block.instructions:
            if (
                isinstance(inst, ir.StoreInstr)
                and inst.operands[1] == return_value
            ):
                remove_list.append(inst)

        # Remove all stores
        for to_remove in remove_list:
            block.instructions.remove(to_remove)

    # Replace non-void return type with void return type and remove return
    # value

    if isinstance(kernel.type, ir.PointerType):
        new_type = ir.PointerType(
            ir.FunctionType(ir.VoidType(), kernel.type.pointee.args[1:])
        )
    else:
        new_type = ir.FunctionType(ir.VoidType(), kernel.type.args[1:])

    kernel.type = new_type
    kernel.return_value = ir.ReturnValue(kernel, ir.VoidType())
    kernel.args = kernel.args[1:]

    # If debug metadata is present, remove the return value from it

    if kernel_metadata := getattr(kernel, "metadata", None):
        if dbg_metadata := kernel_metadata.get("dbg", None):
            for name, value in dbg_metadata.operands:
                if name == "type":
                    type_metadata = value
                    for tm_name, tm_value in type_metadata.operands:
                        if tm_name == "types":
                            types = tm_value
                            types.operands = types.operands[1:]
                            if config.DUMP_LLVM:
                                types._clear_string_cache()

    # Mark as a kernel for NVVM

    nvvm.set_cuda_kernel(kernel)

    if config.DUMP_LLVM:
        print(f"LLVM DUMP: Post kernel fixup {kernel.name}".center(80, "-"))
        print(kernel.module)
        print("=" * 80)


def add_exception_store_helper(kernel):
    # Create global variables for exception state

    def define_error_gv(postfix):
        name = kernel.name + postfix
        gv = cgutils.add_global_variable(kernel.module, ir.IntType(32), name)
        gv.initializer = ir.Constant(gv.type.pointee, None)
        return gv

    gv_exc = define_error_gv("__errcode__")
    gv_tid = []
    gv_ctaid = []
    for i in "xyz":
        gv_tid.append(define_error_gv("__tid%s__" % i))
        gv_ctaid.append(define_error_gv("__ctaid%s__" % i))

    # Create exception store helper function

    helper_name = kernel.name + "__exc_helper__"
    helper_type = ir.FunctionType(ir.VoidType(), (ir.IntType(32),))
    helper_func = ir.Function(kernel.module, helper_type, helper_name)

    block = helper_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Implement status check / exception store logic

    status_code = helper_func.args[0]
    call_conv = cuda_target.target_context.call_conv
    status = call_conv._get_return_status(builder, status_code)

    # Check error status
    with cgutils.if_likely(builder, status.is_ok):
        builder.ret_void()

    with builder.if_then(builder.not_(status.is_python_exc)):
        # User exception raised
        old = ir.Constant(gv_exc.type.pointee, None)

        # Use atomic cmpxchg to prevent rewriting the error status
        # Only the first error is recorded

        xchg = builder.cmpxchg(
            gv_exc, old, status.code, "monotonic", "monotonic"
        )
        changed = builder.extract_value(xchg, 1)

        # If the xchange is successful, save the thread ID.
        sreg = nvvmutils.SRegBuilder(builder)
        with builder.if_then(changed):
            for (
                dim,
                ptr,
            ) in zip("xyz", gv_tid):
                val = sreg.tid(dim)
                builder.store(val, ptr)

            for (
                dim,
                ptr,
            ) in zip("xyz", gv_ctaid):
                val = sreg.ctaid(dim)
                builder.store(val, ptr)

    builder.ret_void()

    return helper_func


@global_compiler_lock
def compile(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=True,
    fastmath=False,
    cc=None,
    opt=None,
    abi="c",
    abi_info=None,
    output="ptx",
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX or LTO-IR for a given set of argument
    types.

    :param pyfunc: The Python function to compile.
    :param sig: The signature representing the function's input and output
                types. If this is a tuple of argument types without a return
                type, the inferred return type is returned by this function. If
                a signature including a return type is passed, the compiled code
                will include a cast from the inferred return type to the
                specified return type, and this function will return the
                specified return type.
    :param debug: Whether to include debug info in the compiled code.
    :type debug: bool
    :param lineinfo: Whether to include a line mapping from the compiled code
                     to the source code. Usually this is used with optimized
                     code (since debug mode would automatically include this),
                     so we want debug info in the LLVM IR but only the line
                     mapping in the final output.
    :type lineinfo: bool
    :param device: Whether to compile a device function.
    :type device: bool
    :param fastmath: Whether to enable fast math flags (ftz=1, prec_sqrt=0,
                     prec_div=, and fma=1)
    :type fastmath: bool
    :param cc: Compute capability to compile for, as a tuple
               ``(MAJOR, MINOR)``. Defaults to ``(5, 0)``.
    :type cc: tuple
    :param opt: Whether to enable optimizations in the compiled code.
    :type opt: bool
    :param abi: The ABI for a compiled function - either ``"numba"`` or
                ``"c"``. Note that the Numba ABI is not considered stable.
                The C ABI is only supported for device functions at present.
    :type abi: str
    :param abi_info: A dict of ABI-specific options. The ``"c"`` ABI supports
                     one option, ``"abi_name"``, for providing the wrapper
                     function's name. The ``"numba"`` ABI has no options.
    :type abi_info: dict
    :param output: Type of output to generate, either ``"ptx"`` or ``"ltoir"``.
    :type output: str
    :param forceinline: Enables inlining at the NVVM IR level when set to
                        ``True``. This is accomplished by adding the
                        ``alwaysinline`` function attribute to the function
                        definition. This is only valid when the output is
                        ``"ltoir"``.
    :param launch_bounds: Kernel launch bounds, specified as a scalar or a tuple
                          of between one and three items. Tuple items provide:

                          - The maximum number of threads per block,
                          - The minimum number of blocks per SM,
                          - The maximum number of blocks per cluster.

                          If a scalar is provided, it is used as the maximum
                          number of threads per block.
    :type launch_bounds: int | tuple[int]
    :return: (code, resty): The compiled code and inferred return type
    :rtype: tuple
    """
    if abi not in ("numba", "c"):
        raise NotImplementedError(f"Unsupported ABI: {abi}")

    if abi == "c" and not device:
        raise NotImplementedError("The C ABI is not supported for kernels")

    if output not in ("ptx", "ltoir"):
        raise NotImplementedError(f"Unsupported output type: {output}")

    if forceinline and not device:
        raise ValueError("Cannot force-inline kernels")

    if forceinline and output != "ltoir":
        raise ValueError("Can only designate forced inlining in LTO-IR")

    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug
    opt = (config.OPT != 0) if opt is None else opt

    if debug and opt:
        msg = (
            "debug=True with opt=True "
            "is not supported by CUDA. This may result in a crash"
            " - set debug=False or opt=False."
        )
        warn(NumbaInvalidConfigWarning(msg))

    lto = output == "ltoir"
    abi_info = abi_info or dict()

    nvvm_options = {"fastmath": fastmath, "opt": 3 if opt else 0}

    if debug:
        nvvm_options["g"] = None

    if lto:
        nvvm_options["gen-lto"] = None

    args, return_type = sigutils.normalize_signature(sig)

    # If the user has used the config variable to specify a non-default that is
    # greater than the lowest non-deprecated one, then we should default to
    # their specified CC instead of the lowest non-deprecated one.
    MIN_CC = max(config.CUDA_DEFAULT_PTX_CC, nvrtc.get_lowest_supported_cc())
    cc = cc or MIN_CC

    cres = compile_cuda(
        pyfunc,
        return_type,
        args,
        debug=debug,
        lineinfo=lineinfo,
        fastmath=fastmath,
        nvvm_options=nvvm_options,
        cc=cc,
        forceinline=forceinline,
    )
    resty = cres.signature.return_type

    if resty and not device and resty != types.void:
        raise TypeError("CUDA kernel must have void return type.")

    tgt = cres.target_context

    if device:
        lib = cres.library
        if abi == "c":
            wrapper_name = abi_info.get("abi_name", pyfunc.__name__)
            lib = cabi_wrap_function(
                tgt, lib, cres.fndesc, wrapper_name, nvvm_options
            )
    else:
        lib = cres.library
        kernel = lib.get_function(cres.fndesc.llvm_func_name)
        lib._entry_name = cres.fndesc.llvm_func_name
        kernel_fixup(kernel, debug)
        nvvm.set_launch_bounds(kernel, launch_bounds)

    if lto:
        code = lib.get_ltoir(cc=cc)
    else:
        code = lib.get_asm_str(cc=cc)
    return code, resty


def compile_for_current_device(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=True,
    fastmath=False,
    opt=None,
    abi="c",
    abi_info=None,
    output="ptx",
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX or LTO-IR for a given signature for the
    current device's compute capabilility. This calls :func:`compile` with an
    appropriate ``cc`` value for the current device."""
    cc = get_current_device().compute_capability
    return compile(
        pyfunc,
        sig,
        debug=debug,
        lineinfo=lineinfo,
        device=device,
        fastmath=fastmath,
        cc=cc,
        opt=opt,
        abi=abi,
        abi_info=abi_info,
        output=output,
        forceinline=forceinline,
        launch_bounds=launch_bounds,
    )


def compile_ptx(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=False,
    fastmath=False,
    cc=None,
    opt=None,
    abi="numba",
    abi_info=None,
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX for a given signature. See
    :func:`compile`. The defaults for this function are to compile a kernel
    with the Numba ABI, rather than :func:`compile`'s default of compiling a
    device function with the C ABI."""
    return compile(
        pyfunc,
        sig,
        debug=debug,
        lineinfo=lineinfo,
        device=device,
        fastmath=fastmath,
        cc=cc,
        opt=opt,
        abi=abi,
        abi_info=abi_info,
        output="ptx",
        forceinline=forceinline,
        launch_bounds=launch_bounds,
    )


def compile_ptx_for_current_device(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=False,
    fastmath=False,
    opt=None,
    abi="numba",
    abi_info=None,
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX for a given signature for the current
    device's compute capabilility. See :func:`compile_ptx`."""
    cc = get_current_device().compute_capability
    return compile_ptx(
        pyfunc,
        sig,
        debug=debug,
        lineinfo=lineinfo,
        device=device,
        fastmath=fastmath,
        cc=cc,
        opt=opt,
        abi=abi,
        abi_info=abi_info,
        forceinline=forceinline,
        launch_bounds=launch_bounds,
    )


def declare_device_function(name, restype, argtypes, link, use_cooperative):
    from .descriptor import cuda_target

    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context
    sig = typing.signature(restype, *argtypes)

    # extfn is the descriptor used to call the function from Python code, and
    # is used as the key for typing and lowering.
    extfn = ExternFunction(name, sig)

    # Typing
    device_function_template = typing.make_concrete_template(name, extfn, [sig])
    typingctx.insert_user_function(extfn, device_function_template)

    # Lowering
    lib = ExternalCodeLibrary(f"{name}_externals", targetctx.codegen())
    for file in link:
        lib.add_linking_file(file)
    lib.use_cooperative = use_cooperative

    # ExternalFunctionDescriptor provides a lowering implementation for calling
    # external functions
    fndesc = funcdesc.ExternalFunctionDescriptor(name, restype, argtypes)
    targetctx.insert_user_function(extfn, fndesc, libs=(lib,))

    return device_function_template


class ExternFunction:
    """A descriptor that can be used to call the external function from within
    a Python kernel."""

    def __init__(self, name, sig):
        self.name = name
        self.sig = sig
