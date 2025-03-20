import numpy as np

from numba.core import types
from numba.core.typeconv.rules import default_casting_rules
from numba.np import numpy_support


class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """
    def __init__(self):
        super().__init__(name='Dim3')


class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """
    def __init__(self):
        super().__init__(name='GridGroup')


dim3 = Dim3()
grid_group = GridGroup()

tid = types.Integer('thread_idx', 32, signed=False)
default_casting_rules.promote_unsafe(tid, types.int64)

# We need to patch the as_dtype function, because it doesn't know how to
# translate the tid type to a NumPy dtype, and there's no way to augment the
# lookup.

_original_as_dtype = numpy_support.as_dtype


def _as_dtype(nbtype):
    nbtype = types.unliteral(nbtype)
    if nbtype == tid:
        return np.dtype('uint32')
    return _original_as_dtype(nbtype)


numpy_support.as_dtype = _as_dtype


class CUDADispatcher(types.Dispatcher):
    """The type of CUDA dispatchers"""
    # This type exists (instead of using types.Dispatcher as the type of CUDA
    # dispatchers) so that we can have an alternative lowering for them to the
    # lowering of CPU dispatchers - the CPU target lowers all dispatchers as a
    # constant address, but we need to lower to a dummy value because it's not
    # generally valid to use the address of CUDA kernels and functions.
    #
    # Notes: it may be a bug in the CPU target that it lowers all dispatchers to
    # a constant address - it should perhaps only lower dispatchers acting as
    # first-class functions to a constant address. Even if that bug is fixed, it
    # is still probably a good idea to have a separate type for CUDA
    # dispatchers, and this type might get other differentiation from the CPU
    # dispatcher type in future.
