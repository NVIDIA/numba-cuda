from numba.core import cgutils, types

from llvmlite import ir

int32_t = ir.IntType(32)
int64_t = ir.IntType(64)
errcode_t = int32_t


def _const_int(code):
    return ir.Constant(errcode_t, code)


RETCODE_OK = _const_int(0)
RETCODE_EXC = _const_int(-1)
RETCODE_NONE = _const_int(-2)
# StopIteration
RETCODE_STOPIT = _const_int(-3)

FIRST_USEREXC = 1

RETCODE_USEREXC = _const_int(FIRST_USEREXC)


class BaseCallConv(object):
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
