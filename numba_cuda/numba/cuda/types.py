from numba.core import types
from numba.core.typeconv import Conversion


class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """

    def __init__(self):
        super().__init__(name="Dim3")


class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """

    def __init__(self):
        super().__init__(name="GridGroup")


dim3 = Dim3()
grid_group = GridGroup()


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


class Bfloat16(types.Number):
    """
    A bfloat16 type.
    """

    def __init__(self):
        super().__init__(name="__nv_bfloat16")

        self.alignof_ = 2
        self.bitwidth = 16

    def can_convert_from(self, other):
        if isinstance(other, types.Float):
            return Conversion.unsafe

        elif isinstance(other, types.Integer):
            if other.bitwidth == 8:
                return Conversion.safe

        return Conversion.unsafe

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.Float):
            if other.bitwidth >= 32:
                return Conversion.safe
            else:
                return Conversion.unsafe
        elif isinstance(other, types.Integer):
            return Conversion.unsafe

        return Conversion.unsafe

    def unify(self, typingctx, other):
        if isinstance(other, (types.Float, types.Integer)):
            return typingctx.unify_pairs(self, other)

    def cast_python_value(self, value):
        try:
            import ml_dtypes  # noqa: F401

            return ml_dtypes.bfloat16(value)
        except ImportError:
            raise NotImplementedError


bfloat16 = Bfloat16()
