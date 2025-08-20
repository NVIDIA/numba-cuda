from numba.core import types
from numba.cuda.cudadrv import nvvm


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


class CUDAArray(types.Array):
    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=nvvm.ADDRSPACE_GENERIC,
    ):
        super().__init__(dtype, ndim, layout, readonly, name, aligned)
        self.addrspace = addrspace

        if name is None:
            type_name = "array"
            if not self.mutable:
                type_name = "readonly " + type_name
            if not self.aligned:
                type_name = "unaligned " + type_name
            self.name = "%s(%s, %sd, %s, addrspace(%d))" % (
                type_name,
                dtype,
                ndim,
                layout,
                addrspace,
            )

    def __repr__(self):
        return (
            f"CUDAArray({repr(self.dtype)}, {self.ndim}, '{self.layout}', "
            f"{not self.mutable}, aligned={self.aligned}, addrspace={self.addrspace})"
        )

    @property
    def key(self):
        return *super().key, self.addrspace

    def copy(
        self,
        dtype=None,
        ndim=None,
        layout=None,
        readonly=None,
        addrspace=None,
    ):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        if addrspace is None:
            addrspace = self.addrspace
        return type(self)(
            dtype=dtype,
            ndim=ndim,
            layout=layout,
            readonly=readonly,
            aligned=self.aligned,
            addrspace=addrspace,
        )

    @classmethod
    def from_array_type(cls, ary, addrspace=nvvm.ADDRSPACE_GENERIC):
        """
        Create a CUDAArray type from a numpy array.
        """
        return cls(
            dtype=ary.dtype,
            ndim=ary.ndim,
            layout=ary.layout,
            readonly=not ary.mutable,
            aligned=ary.aligned,
            addrspace=addrspace,
        )
