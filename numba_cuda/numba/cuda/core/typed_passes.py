import abc
import warnings
from contextlib import contextmanager
from numba.core import errors, types, funcdesc
from numba.core.compiler_machinery import LoweringPass
from llvmlite import binding as llvm


@contextmanager
def fallback_context(state, msg):
    """
    Wraps code that would signal a fallback to object mode
    """
    try:
        yield
    except Exception as e:
        if not state.status.can_fallback:
            raise
        else:
            # Clear all references attached to the traceback
            e = e.with_traceback(None)
            # this emits a warning containing the error message body in the
            # case of fallback from npm to objmode
            loop_lift = "" if state.flags.enable_looplift else "OUT"
            msg_rewrite = (
                "\nCompilation is falling back to object mode "
                "WITH%s looplifting enabled because %s" % (loop_lift, msg)
            )
            warnings.warn_explicit(
                "%s due to: %s" % (msg_rewrite, e),
                errors.NumbaWarning,
                state.func_id.filename,
                state.func_id.firstlineno,
            )
            raise


class BaseNativeLowering(abc.ABC, LoweringPass):
    """The base class for a lowering pass. The lowering functionality must be
    specified in inheriting classes by providing an appropriate lowering class
    implementation in the overridden `lowering_class` property."""

    _name = None

    def __init__(self):
        LoweringPass.__init__(self)

    @property
    @abc.abstractmethod
    def lowering_class(self):
        """Returns the class that performs the lowering of the IR describing the
        function that is the target of the current compilation."""
        pass

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        library = state.library
        targetctx = state.targetctx
        interp = state.func_ir  # why is it called this?!
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata
        pre_stats = llvm.passmanagers.dump_refprune_stats()

        msg = "Function %s failed at nopython mode lowering" % (
            state.func_id.func_name,
        )
        with fallback_context(state, msg):
            # Lowering
            fndesc = (
                funcdesc.PythonFunctionDescriptor.from_specialized_function(
                    interp,
                    typemap,
                    restype,
                    calltypes,
                    mangler=targetctx.mangler,
                    inline=flags.forceinline,
                    noalias=flags.noalias,
                    abi_tags=[flags.get_mangle_string()],
                )
            )

            with targetctx.push_code_library(library):
                lower = self.lowering_class(
                    targetctx, library, fndesc, interp, metadata=metadata
                )
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)

                if not flags.no_cfunc_wrapper:
                    # skip cfunc wrapper generation if unsupported
                    # argument or return types are used
                    for t in state.args:
                        if isinstance(t, (types.Omitted, types.Generator)):
                            break
                    else:
                        if isinstance(
                            restype, (types.Optional, types.Generator)
                        ):
                            pass
                        else:
                            lower.create_cfunc_wrapper()

                env = lower.env
                call_helper = lower.call_helper
                del lower

            from numba.core.compiler import _LowerResult  # TODO: move this

            if flags.no_compile:
                state["cr"] = _LowerResult(
                    fndesc, call_helper, cfunc=None, env=env
                )
            else:
                # Prepare for execution
                # Insert native function for use by other jitted-functions.
                # We also register its library to allow for inlining.
                cfunc = targetctx.get_executable(library, fndesc, env)
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state["cr"] = _LowerResult(
                    fndesc, call_helper, cfunc=cfunc, env=env
                )

            # capture pruning stats
            post_stats = llvm.passmanagers.dump_refprune_stats()
            metadata["prune_stats"] = post_stats - pre_stats

            # Save the LLVM pass timings
            metadata["llvm_pass_timings"] = library.recorded_timings
        return True
