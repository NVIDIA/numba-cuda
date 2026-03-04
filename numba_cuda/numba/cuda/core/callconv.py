# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda import types
from numba.cuda import cgutils
from numba.cuda import itanium_mangler
from numba.cuda.core import imputils
from collections import namedtuple

from llvmlite import ir

int32_t = ir.IntType(32)
int64_t = ir.IntType(64)
errcode_t = int32_t


Status = namedtuple(
    "Status",
    (
        "code",
        # If the function returned ok (a value or None)
        "is_ok",
        # If the function returned None
        "is_none",
        # If the function errored out (== not is_ok)
        "is_error",
        # If the generator exited with StopIteration
        "is_stop_iteration",
        # If the function errored with an already set exception
        "is_python_exc",
        # If the function errored with a user exception
        "is_user_exc",
        # The pointer to the exception info structure (for user
        # exceptions)
        "excinfoptr",
    ),
)


def _const_int(code):
    return ir.Constant(errcode_t, code)


RETCODE_OK = _const_int(0)
RETCODE_EXC = _const_int(-1)
RETCODE_NONE = _const_int(-2)
# StopIteration
RETCODE_STOPIT = _const_int(-3)

FIRST_USEREXC = 1

RETCODE_USEREXC = _const_int(FIRST_USEREXC)


class BaseCallConv:
    def __init__(self, context):
        self.context = context

    def return_optional_value(self, builder, retty, valty, value):
        if valty == types.none:
            # Value is none
            self.return_native_none(builder)

        elif retty == valty:
            # Value is an optional, need a runtime switch
            optval = self.context.make_helper(builder, retty, value=value)

            validbit = cgutils.as_bool_bit(builder, optval.valid)
            with builder.if_then(validbit):
                retval = self.context.get_return_value(
                    builder, retty.type, optval.data
                )
                self.return_value(builder, retval)

            self.return_native_none(builder)

        elif not isinstance(valty, types.Optional):
            # Value is not an optional, need a cast
            if valty != retty.type:
                value = self.context.cast(
                    builder, value, fromty=valty, toty=retty.type
                )
            retval = self.context.get_return_value(builder, retty.type, value)
            self.return_value(builder, retval)

        else:
            raise NotImplementedError(
                "returning {0} for {1}".format(valty, retty)
            )

    def return_native_none(self, builder):
        self._return_errcode_raw(builder, RETCODE_NONE)

    def return_exc(self, builder):
        self._return_errcode_raw(builder, RETCODE_EXC)

    def return_stop_iteration(self, builder):
        self._return_errcode_raw(builder, RETCODE_STOPIT)

    def get_return_type(self, ty):
        """
        Get the actual type of the return argument for Numba type *ty*.
        """
        restype = self.context.data_model_manager[ty].get_return_type()
        return restype.as_pointer()

    def init_call_helper(self, builder):
        """
        Initialize and return a call helper object for the given builder.
        """
        ch = self._make_call_helper(builder)
        builder.__call_helper = ch
        return ch

    def _get_call_helper(self, builder):
        return builder.__call_helper

    def unpack_exception(self, builder, pyapi, status):
        return pyapi.unserialize(status.excinfoptr)

    def raise_error(self, builder, pyapi, status):
        """
        Given a non-ok *status*, raise the corresponding Python exception.
        """
        bbend = builder.function.append_basic_block()

        with builder.if_then(status.is_user_exc):
            # Unserialize user exception.
            # Make sure another error may not interfere.
            pyapi.err_clear()
            exc = self.unpack_exception(builder, pyapi, status)
            with cgutils.if_likely(builder, cgutils.is_not_null(builder, exc)):
                pyapi.raise_object(exc)  # steals ref
            builder.branch(bbend)

        with builder.if_then(status.is_stop_iteration):
            pyapi.err_set_none("PyExc_StopIteration")
            builder.branch(bbend)

        with builder.if_then(status.is_python_exc):
            # Error already raised => nothing to do
            builder.branch(bbend)

        pyapi.err_set_string(
            "PyExc_SystemError", "unknown error when calling native function"
        )
        builder.branch(bbend)

        builder.position_at_end(bbend)

    def decode_arguments(self, builder, argtypes, func):
        """
        Get the decoded (unpacked) Python arguments with *argtypes*
        from LLVM function *func*.  A tuple of LLVM values is returned.
        """
        raw_args = self.get_arguments(func)
        arginfo = self._get_arg_packer(argtypes)
        return arginfo.from_arguments(builder, raw_args)

    def _get_arg_packer(self, argtypes):
        """
        Get an argument packer for the given argument types.
        """
        return self.context.get_arg_packer(argtypes)

    def mangler(self, name, argtypes, *, abi_tags=(), uid=None):
        return itanium_mangler.mangle(
            name, argtypes, abi_tags=abi_tags, uid=uid
        )

    def call_internal_no_propagate(self, builder, fndesc, sig, args):
        """Similar to `.call_internal()` but does not handle or propagate
        the return status automatically.
        """
        llvm_mod = builder.module
        fn = fndesc.declare_function(llvm_mod)
        # Marshal the call using the callee's ABI.
        status, res = fndesc.call_conv.call_function(
            builder, fn, sig.return_type, sig.args, args
        )
        return status, res


class MinimalCallConv(BaseCallConv):
    """
    A minimal calling convention, suitable for e.g. GPU targets.
    The implemented function signature is:

        retcode_t (<Python return type>*, ... <Python arguments>)

    The return code will be one of the RETCODE_* constants or a
    function-specific user exception id (>= RETCODE_USEREXC).

    Caller is responsible for allocating a slot for the return value
    (passed as a pointer in the first argument).
    """

    def _make_call_helper(self, builder):
        return _MinimalCallHelper()

    def return_value(self, builder, retval):
        retptr = builder.function.args[0]
        assert retval.type == retptr.type.pointee, (
            str(retval.type),
            str(retptr.type.pointee),
        )
        builder.store(retval, retptr)
        self._return_errcode_raw(builder, RETCODE_OK)

    def return_user_exc(
        self, builder, exc, exc_args=None, loc=None, func_name=None
    ):
        if exc is not None and not issubclass(exc, BaseException):
            raise TypeError(
                "exc should be None or exception class, got %r" % (exc,)
            )
        if exc_args is not None and not isinstance(exc_args, tuple):
            raise TypeError(
                "exc_args should be None or tuple, got %r" % (exc_args,)
            )

        # Build excinfo struct
        if loc is not None:
            fname = loc._raw_function_name()
            if fname is None:
                # could be exec(<string>) or REPL, try func_name
                fname = func_name

            locinfo = (fname, loc.filename, loc.line)
            if None in locinfo:
                locinfo = None
        else:
            locinfo = None

        call_helper = self._get_call_helper(builder)
        exc_id = call_helper._add_exception(exc, exc_args, locinfo)
        self._return_errcode_raw(builder, _const_int(exc_id))

    def return_status_propagate(self, builder, status):
        self._return_errcode_raw(builder, status.code)

    def _return_errcode_raw(self, builder, code):
        if isinstance(code, int):
            code = _const_int(code)
        builder.ret(code)

    def _get_return_status(self, builder, code):
        """
        Given a return *code*, get a Status instance.
        """
        norm = builder.icmp_signed("==", code, RETCODE_OK)
        none = builder.icmp_signed("==", code, RETCODE_NONE)
        ok = builder.or_(norm, none)
        err = builder.not_(ok)
        exc = builder.icmp_signed("==", code, RETCODE_EXC)
        is_stop_iteration = builder.icmp_signed("==", code, RETCODE_STOPIT)
        is_user_exc = builder.icmp_signed(">=", code, RETCODE_USEREXC)

        status = Status(
            code=code,
            is_ok=ok,
            is_error=err,
            is_python_exc=exc,
            is_none=none,
            is_user_exc=is_user_exc,
            is_stop_iteration=is_stop_iteration,
            excinfoptr=None,
        )
        return status

    def get_function_type(self, restype, argtypes):
        """
        Get the implemented Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        resptr = self.get_return_type(restype)
        fnty = ir.FunctionType(errcode_t, [resptr] + argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn), ["arg." + a for a in args])
        fn.args[0].name = ".ret"

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args[1:]

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(cgutils.get_null_value(retty), retvaltmp)

        arginfo = self._get_arg_packer(argtys)
        args = arginfo.as_arguments(builder, args)
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out

    def call_internal(self, builder, fndesc, sig, args):
        """
        Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
        status, res = self.call_internal_no_propagate(
            builder, fndesc, sig, args
        )
        if status is not None:
            with cgutils.if_unlikely(builder, status.is_error):
                self.return_status_propagate(builder, status)

        res = imputils.fix_returning_optional(
            self.context, builder, sig, status, res
        )
        return res


class _MinimalCallHelper:
    """
    A call helper object for the "minimal" calling convention.
    User exceptions are represented as integer codes and stored in
    a mapping for retrieval from the caller.
    """

    def __init__(self):
        self.exceptions = {}

    def _add_exception(self, exc, exc_args, locinfo):
        """
        Add a new user exception to this helper. Returns an integer that can be
        used to refer to the added exception in future.

        Parameters
        ----------
        exc :
            exception type
        exc_args : None or tuple
            exception args
        locinfo : tuple
            location information
        """
        exc_id = len(self.exceptions) + FIRST_USEREXC
        self.exceptions[exc_id] = exc, exc_args, locinfo
        return exc_id

    def get_exception(self, exc_id):
        """
        Get information about a user exception. Returns a tuple of
        (exception type, exception args, location information).

        Parameters
        ----------
        id : integer
            The ID of the exception to look up
        """
        try:
            return self.exceptions[exc_id]
        except KeyError:
            msg = "unknown error %d in native function" % exc_id
            exc = SystemError
            exc_args = (msg,)
            locinfo = None
            return exc, exc_args, locinfo


class CUDACallConv(MinimalCallConv):
    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        # Do not prefix "arg." on argument name, so that nvvm compiler
        # can track debug info of argument more accurately
        arginfo.assign_names(self.get_arguments(fn), args)
        fn.args[0].name = ".ret"


class CUDACABICallConv(BaseCallConv):
    """
    Calling convention aimed at matching the CUDA C/C++ ABI. The implemented
    function signature is:

        <Python return type> (<Python arguments>)

    Exceptions are unsupported in this convention.
    """

    def _make_call_helper(self, builder):
        # Call helpers are used to help report exceptions back to Python, so
        # none is required here.
        return None

    def return_value(self, builder, retval):
        expected_type = builder.function.ftype.return_type
        actual_type = retval.type

        # If types don't match, we need to cast
        if actual_type != expected_type:
            if isinstance(actual_type, ir.IntType) and isinstance(
                expected_type, ir.IntType
            ):
                if actual_type.width < expected_type.width:
                    # Zero-extend smaller integers to larger ones
                    retval = builder.zext(retval, expected_type)
                elif actual_type.width > expected_type.width:
                    # Truncate larger integers to smaller ones
                    retval = builder.trunc(retval, expected_type)

        return builder.ret(retval)

    def return_user_exc(
        self, builder, exc, exc_args=None, loc=None, func_name=None
    ):
        # C ABI has no status channel to propagate Python exceptions.
        return

    def return_status_propagate(self, builder, status):
        # C ABI has no status channel to propagate lower-frame failures.
        return

    def get_function_type(self, restype, argtypes):
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        fnty = ir.FunctionType(self.get_return_type(restype), argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn), ["arg." + a for a in args])

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        arginfo = self._get_arg_packer(argtys)
        realargs = arginfo.as_arguments(builder, args)
        code = builder.call(callee, realargs)
        # No status required as we don't support exceptions or a distinct None
        # value in a C ABI.
        status = None
        out = self.context.get_returned_value(builder, resty, code)
        return status, out

    def call_internal(self, builder, fndesc, sig, args):
        """
        Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
        status, res = self.call_internal_no_propagate(
            builder, fndesc, sig, args
        )

        # CABI intentionally ignores lower-frame error codes.
        if not isinstance(sig.return_type, types.Optional):
            return res

        # A callee without a status channel cannot represent None.
        if status is None:
            return res

        # Flatten Optional[T] into plain T for CABI:
        # - if value is present, return it
        # - if value is None, return a default-initialized T
        value_type = sig.return_type.type
        default_value = self.context.get_constant_null(value_type)

        outptr = cgutils.alloca_once_value(builder, default_value)
        with builder.if_then(builder.not_(status.is_none)):
            builder.store(res, outptr)
        return builder.load(outptr)

    def get_return_type(self, ty):
        return self.context.data_model_manager[ty].get_return_type()

    def mangler(self, name, argtypes, *, abi_tags=None, uid=None):
        if name.startswith(".NumbaEnv."):
            func_name = name.split(".")[-1]
            return f"_ZN08NumbaEnv{func_name}"
        return name.split(".")[-1]


class ErrorModel:
    def __init__(self, call_conv):
        self.call_conv = call_conv

    def fp_zero_division(self, builder, exc_args=None, loc=None):
        if self.raise_on_fp_zero_division:
            self.call_conv.return_user_exc(
                builder,
                ZeroDivisionError,
                exc_args=exc_args,
                loc=loc,
            )
            return True
        return False


class PythonErrorModel(ErrorModel):
    """
    The Python error model.  Any invalid FP input raises an exception.
    """

    raise_on_fp_zero_division = True


class NumpyErrorModel(ErrorModel):
    """
    In the Numpy error model, floating-point errors don't raise an
    exception.  The FPU exception state is inspected by Numpy at the
    end of a ufunc's execution and a warning is raised if appropriate.

    Note there's no easy way to set the FPU exception state from LLVM.
    Instructions known to set an FP exception can be optimized away:
        https://llvm.org/bugs/show_bug.cgi?id=6050
        http://lists.llvm.org/pipermail/llvm-dev/2014-September/076918.html
        http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140929/237997.html
    """

    raise_on_fp_zero_division = False


error_models = {
    "python": PythonErrorModel,
    "numpy": NumpyErrorModel,
}


def create_error_model(model_name, call_conv):
    """
    Create an error model instance for the given target context.
    """
    return error_models[model_name](call_conv)
