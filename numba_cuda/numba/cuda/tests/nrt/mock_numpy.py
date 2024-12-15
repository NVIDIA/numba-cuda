import math

import numpy as np

from numba.core import errors, types
from numba.core.extending import overload
from numba.np.arrayobj import (_check_const_str_dtype, is_nonelike,
                               ty_parse_dtype, ty_parse_shape, numpy_empty_nd,
                               numpy_empty_like_nd)


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


def cuda_empty_like(arr):
    pass


def cuda_arange(start):
    pass


def cuda_ones(shape):
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


@overload(cuda_empty_like)
def ol_cuda_empty_like(arr):

    if isinstance(arr, types.Array):
        nb_dtype = arr.dtype
    else:
        nb_dtype = arr

    if isinstance(arr, types.Array):
        layout = arr.layout if arr.layout != 'A' else 'C'
        retty = arr.copy(dtype=nb_dtype, layout=layout, readonly=False)
    else:
        retty = types.Array(nb_dtype, 0, 'C')

    def impl(arr):
        dtype = None
        return numpy_empty_like_nd(arr, dtype, retty)
    return impl


def _arange_dtype(*args):
    bounds = [a for a in args if not isinstance(a, types.NoneType)]

    if any(isinstance(a, types.Complex) for a in bounds):
        dtype = types.complex128
    elif any(isinstance(a, types.Float) for a in bounds):
        dtype = types.float64
    else:
        # `np.arange(10).dtype` is always `np.dtype(int)`, aka `np.int_`, which
        # in all released versions of numpy corresponds to the C `long` type.
        # Windows 64 is broken by default here because Numba (as of 0.47) does
        # not differentiate between Python and NumPy integers, so a `typeof(1)`
        # on w64 is `int64`, i.e. `intp`. This means an arange(<some int>) will
        # be typed as arange(int64) and the following will yield int64 opposed
        # to int32. Example: without a load of analysis to work out of the args
        # were wrapped in NumPy int*() calls it's not possible to detect the
        # difference between `np.arange(10)` and `np.arange(np.int64(10)`.
        NPY_TY = getattr(types, "int%s" % (8 * np.dtype(int).itemsize))

        # unliteral these types such that `max` works.
        unliteral_bounds = [types.unliteral(x) for x in bounds]
        dtype = max(unliteral_bounds + [NPY_TY,])

    return dtype


@overload(cuda_arange)
def ol_cuda_arange(start):
    """Simplified arange with just 1 argument."""
    if (not isinstance(start, types.Number)):
        return

    start_value = getattr(start, "literal_value", None)

    def impl(start):
        # Allow for improved performance if given literal arguments.
        lit_start = start_value if start_value is not None else start

        _step = 1
        _start, _stop = 0, lit_start

        nitems_c = (_stop - _start) / _step
        nitems_r = int(math.ceil(nitems_c.real))

        # Binary operator needed for compiler branch pruning.
        nitems = max(nitems_r, 0)

        arr = cuda_empty(nitems, np.int64)
        val = _start
        for i in range(nitems):
            arr[i] = val + (i * _step)
        return arr

    return impl


@overload(cuda_ones)
def ol_cuda_ones(shape):

    def impl(shape):
        arr = cuda_empty(shape, np.float64)
        arr_flat = arr.flat
        for idx in range(len(arr_flat)):
            arr_flat[idx] = 1
        return arr
    return impl
