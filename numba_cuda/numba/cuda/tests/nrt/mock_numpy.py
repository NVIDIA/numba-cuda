
from numba.core import errors, types
from numba.core.extending import overload
from numba.np.arrayobj import (_check_const_str_dtype, is_nonelike,
                               ty_parse_dtype, ty_parse_shape, numpy_empty_nd)


# Typical tests for allocation use array construction (e.g. np.zeros, np.empty,
# etc.) to induce allocations. These don't work in the CUDA target because they
# need keyword arguments, which are presently not supported properly in the
# CUDA target.
#
# To work around this, we can define our own function, that works like
# the desired one, except that it uses only positional arguments.
#
# Once the CUDA target supports keyword arguments, this workaround will no
# longer be necessary and the tests in this module should be switched to use
# the relevant NumPy functions instead.
def cuda_empty(shape, dtype):
    pass


@overload(cuda_empty)
def ol_cuda_empty(shape, dtype):
    _check_const_str_dtype("empty", dtype)
    if (dtype is float or
        (isinstance(dtype, types.Function) and dtype.typing_key is float) or
            is_nonelike(dtype)): #default
        nb_dtype = types.double
    else:
        nb_dtype = ty_parse_dtype(dtype)

    ndim = ty_parse_shape(shape)
    if nb_dtype is not None and ndim is not None:
        retty = types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        def impl(shape, dtype):
            return numpy_empty_nd(shape, dtype, retty)
        return impl
    else:
        msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
        raise errors.TypingError(msg)
