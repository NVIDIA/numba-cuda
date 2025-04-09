"""
Added for symmetry with the core API
"""

from numba.core.extending import intrinsic as _intrinsic
from numba.cuda.models import register_model  # noqa: F401
from numba.cuda import models  # noqa: F401

intrinsic = _intrinsic(target="cuda")


def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.
    """
    from numba.core.typing.templates import AttributeTemplate

    from numba.core.datamodel import default_manager
    from numba.core.datamodel.models import StructModel
    from numba.core.imputils import impl_ret_borrowed
    from numba.core import cgutils, types

    from numba.cuda.models import cuda_data_manager
    from numba.cuda.cudadecl import registry as cuda_registry
    from numba.cuda.cudaimpl import registry as cuda_impl_registry

    data_model_manager = cuda_data_manager.chain(default_manager)

    if not isinstance(typeclass, type) or not issubclass(typeclass, types.Type):
        raise TypeError(f"typeclass should be a Type subclass, got {typeclass}")

    def get_attr_fe_type(typ):
        """
        Get the Numba type of member *struct_attr* in *typ*.
        """
        model = data_model_manager.lookup(typ)
        if not isinstance(model, StructModel):
            raise TypeError(
                f"make_struct_attribute_wrapper() needs a type with a StructModel, but got {model}"
            )
        return model.get_member_fe_type(struct_attr)

    @cuda_registry.register_attr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @cuda_impl_registry.lower_getattr(typeclass, python_attr)
    def struct_getattr_impl(context, builder, typ, val):
        val = cgutils.create_struct_proxy(typ)(context, builder, value=val)
        attrty = get_attr_fe_type(typ)
        attrval = getattr(val, struct_attr)
        return impl_ret_borrowed(context, builder, attrty, attrval)
