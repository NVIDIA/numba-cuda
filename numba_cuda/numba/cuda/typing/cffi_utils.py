"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""

from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.typing import templates

try:
    import cffi

    ffi = cffi.FFI()
except ImportError:
    ffi = None

SUPPORTED = ffi is not None
registry = templates.Registry()


@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = "ffi.from_buffer"

    def generic(self, args, kws):
        if kws or len(args) != 1:
            return
        [ary] = args
        if not isinstance(ary, types.Buffer):
            raise TypingError(
                "from_buffer() expected a buffer object, got %s" % (ary,)
            )
        if ary.layout not in ("C", "F"):
            raise TypingError(
                "from_buffer() unsupported on non-contiguous buffers (got %s)"
                % (ary,)
            )
        if ary.layout != "C" and ary.ndim > 1:
            raise TypingError(
                "from_buffer() only supports multidimensional arrays with C layout (got %s)"
                % (ary,)
            )
        ptr = types.CPointer(ary.dtype)
        return templates.signature(ptr, ary)


@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.ffi

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, types.ffi)
