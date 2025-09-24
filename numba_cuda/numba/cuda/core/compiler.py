# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.core.tracing import event

from numba.cuda.core import bytecode
from numba.core import callconv, errors
from numba.cuda.core import config
from numba.core.errors import CompilerError

from numba.cuda.core.untyped_passes import ExtractByteCode, FixupArgs
from numba.cuda.core.targetconfig import ConfigStack


class _CompileStatus(object):
    """
    Describes the state of compilation. Used like a C record.
    """

    __slots__ = ["fail_reason", "can_fallback"]

    def __init__(self, can_fallback):
        self.fail_reason = None
        self.can_fallback = can_fallback

    def __repr__(self):
        vals = []
        for k in self.__slots__:
            vals.append("{k}={v}".format(k=k, v=getattr(self, k)))
        return ", ".join(vals)


class StateDict(dict):
    """
    A dictionary that has an overloaded getattr and setattr to permit getting
    and setting key/values through the use of attributes.
    """

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


class _EarlyPipelineCompletion(Exception):
    """
    Raised to indicate that a pipeline has completed early
    """

    def __init__(self, result):
        self.result = result


def _make_subtarget(targetctx, flags):
    """
    Make a new target context from the given target context and flags.
    """
    subtargetoptions = {}
    if flags.debuginfo:
        subtargetoptions["enable_debuginfo"] = True
    if flags.boundscheck:
        subtargetoptions["enable_boundscheck"] = True
    if flags.nrt:
        subtargetoptions["enable_nrt"] = True
    if flags.fastmath:
        subtargetoptions["fastmath"] = flags.fastmath
    error_model = callconv.create_error_model(flags.error_model, targetctx)
    subtargetoptions["error_model"] = error_model

    return targetctx.subtarget(**subtargetoptions)


class CompilerBase(object):
    """
    Stores and manages states for the compiler
    """

    def __init__(
        self, typingctx, targetctx, library, args, return_type, flags, locals
    ):
        # Make sure the environment is reloaded
        config.reload_config()
        typingctx.refresh()
        targetctx.refresh()

        self.state = StateDict()

        self.state.typingctx = typingctx
        self.state.targetctx = _make_subtarget(targetctx, flags)
        self.state.library = library
        self.state.args = args
        self.state.return_type = return_type
        self.state.flags = flags
        self.state.locals = locals

        # Results of various steps of the compilation pipeline
        self.state.bc = None
        self.state.func_id = None
        self.state.func_ir = None
        self.state.lifted = None
        self.state.lifted_from = None
        self.state.typemap = None
        self.state.calltypes = None
        self.state.type_annotation = None
        # holds arbitrary inter-pipeline stage meta data
        self.state.metadata = {}
        self.state.reload_init = []
        # hold this for e.g. with_lifting, null out on exit
        self.state.pipeline = self

        self.state.status = _CompileStatus(
            can_fallback=self.state.flags.enable_pyobject
        )

    def compile_extra(self, func):
        self.state.func_id = bytecode.FunctionIdentity.from_function(func)
        ExtractByteCode().run_pass(self.state)

        self.state.lifted = ()
        self.state.lifted_from = None
        return self._compile_bytecode()

    def compile_ir(self, func_ir, lifted=(), lifted_from=None):
        self.state.func_id = func_ir.func_id
        self.state.lifted = lifted
        self.state.lifted_from = lifted_from
        self.state.func_ir = func_ir
        self.state.nargs = self.state.func_ir.arg_count

        FixupArgs().run_pass(self.state)
        return self._compile_ir()

    def define_pipelines(self):
        """Child classes override this to customize the pipelines in use."""
        raise NotImplementedError()

    def _compile_core(self):
        """
        Populate and run compiler pipeline
        """
        with ConfigStack().enter(self.state.flags.copy()):
            pms = self.define_pipelines()
            for pm in pms:
                pipeline_name = pm.pipeline_name
                func_name = "%s.%s" % (
                    self.state.func_id.modname,
                    self.state.func_id.func_qualname,
                )

                event("Pipeline: %s for %s" % (pipeline_name, func_name))
                self.state.metadata["pipeline_times"] = {
                    pipeline_name: pm.exec_times
                }
                is_final_pipeline = pm == pms[-1]
                res = None
                try:
                    pm.run(self.state)
                    if self.state.cr is not None:
                        break
                except _EarlyPipelineCompletion as e:
                    res = e.result
                    break
                except Exception as e:
                    if not isinstance(e, errors.NumbaError):
                        raise e
                    self.state.status.fail_reason = e
                    if is_final_pipeline:
                        raise e
            else:
                raise CompilerError("All available pipelines exhausted")

            # Pipeline is done, remove self reference to release refs to user
            # code
            self.state.pipeline = None

            # organise a return
            if res is not None:
                # Early pipeline completion
                return res
            else:
                assert self.state.cr is not None
                return self.state.cr

    def _compile_bytecode(self):
        """
        Populate and run pipeline for bytecode input
        """
        assert self.state.func_ir is None
        return self._compile_core()

    def _compile_ir(self):
        """
        Populate and run pipeline for IR input
        """
        assert self.state.func_ir is not None
        return self._compile_core()
