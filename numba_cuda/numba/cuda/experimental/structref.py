# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Utilities for defining a mutable struct.

A mutable struct is passed by reference;
hence, structref (a reference to a struct).

"""

from numba.cuda import jit
from numba.core import types, imputils
from numba.cuda import cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
    infer_getattr,
    lower_getattr_generic,
    lower_setattr_generic,
    box,
    unbox,
    NativeValue,
)
from numba.cuda.typing.templates import AttributeTemplate


class _Utils:
    """Internal builder-code utils for structref definitions."""

    def __init__(self, context, builder, struct_type):
        """
        Parameters
        ----------
        context :
            a numba target context
        builder :
            a llvmlite IRBuilder
        struct_type : numba.core.types.StructRef
        """
        self.context = context
        self.builder = builder
        self.struct_type = struct_type

    def new_struct_ref(self, mi):
        """Encapsulate the MemInfo from a `StructRefPayload` in a `StructRef`"""
        context = self.context
        builder = self.builder
        struct_type = self.struct_type

        st = cgutils.create_struct_proxy(struct_type)(context, builder)
        st.meminfo = mi
        return st

    def get_struct_ref(self, val):
        """Return a helper for accessing a StructRefType"""
        context = self.context
        builder = self.builder
        struct_type = self.struct_type

        return cgutils.create_struct_proxy(struct_type)(
            context, builder, value=val
        )

    def get_data_pointer(self, val):
        """Get the data pointer to the payload from a `StructRefType`."""
        context = self.context
        builder = self.builder
        struct_type = self.struct_type

        structval = self.get_struct_ref(val)
        meminfo = structval.meminfo
        data_ptr = context.nrt.meminfo_data(builder, meminfo)

        valtype = struct_type.get_data_type()
        model = context.data_model_manager[valtype]
        alloc_type = model.get_value_type()
        data_ptr = builder.bitcast(data_ptr, alloc_type.as_pointer())
        return data_ptr

    def get_data_struct(self, val):
        """Get a getter/setter helper for accessing a `StructRefPayload`"""
        context = self.context
        builder = self.builder
        struct_type = self.struct_type

        data_ptr = self.get_data_pointer(val)
        valtype = struct_type.get_data_type()
        dataval = cgutils.create_struct_proxy(valtype)(
            context, builder, ref=data_ptr
        )
        return dataval


def define_attributes(struct_typeclass):
    """Define attributes on `struct_typeclass`.

    Defines both setters and getters in jit-code.

    This is called directly in `register()`.
    """

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, typ, attr):
            if attr in typ.field_dict:
                attrty = typ.field_dict[attr]
                return attrty

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, typ, val, attr):
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

    @lower_setattr_generic(struct_typeclass)
    def struct_setattr_impl(context, builder, sig, args, attr):
        [inst_type, val_type] = sig.args
        [instance, val] = args
        utils = _Utils(context, builder, inst_type)
        dataval = utils.get_data_struct(instance)
        # cast val to the correct type
        field_type = inst_type.field_dict[attr]
        casted = context.cast(builder, val, val_type, field_type)
        # read old
        old_value = getattr(dataval, attr)
        # incref new value
        context.nrt.incref(builder, val_type, casted)
        # decref old value (must be last in case new value is old value)
        context.nrt.decref(builder, val_type, old_value)
        # write new
        setattr(dataval, attr, casted)


def define_boxing(struct_type, obj_class):
    """Define the boxing & unboxing logic for `struct_type` to `obj_class`.

    Defines both boxing and unboxing.

    - boxing turns an instance of `struct_type` into a PyObject of `obj_class`
    - unboxing turns an instance of `obj_class` into an instance of
      `struct_type` in jit-code.


    Use this directly instead of `define_proxy()` when the user does not
    want any constructor to be defined.
    """
    if struct_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")

    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_struct_ref(typ, val, c):
        """
        Convert a raw pointer to a Python int.
        """
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.get_struct_ref(val)
        meminfo = struct_ref.meminfo

        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)

        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

        res = c.pyapi.call_function_objargs(
            ctor_pyfunc,
            [ty_pyobj, boxed_meminfo],
        )
        c.pyapi.decref(ctor_pyfunc)
        c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        return res

    @unbox(struct_type)
    def unbox_struct_ref(typ, obj, c):
        mi_obj = c.pyapi.object_getattr_string(obj, "_meminfo")

        mip_type = types.MemInfoPointer(types.voidptr)

        mi = c.unbox(mip_type, mi_obj).value

        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.new_struct_ref(mi)
        out = struct_ref._getvalue()

        c.pyapi.decref(mi_obj)
        return NativeValue(out)


def register(struct_type):
    """Register a `numba.core.types.StructRef` for use in jit-code.

    This defines the data-model for lowering an instance of `struct_type`.
    This defines attributes accessor and mutator for an instance of
    `struct_type`.

    Parameters
    ----------
    struct_type : type
        A subclass of `numba.core.types.StructRef`.

    Returns
    -------
    struct_type : type
        Returns the input argument so this can act like a decorator.

    Examples
    --------

    .. code-block::

        class MyStruct(numba.core.types.StructRef):
            ...  # the simplest subclass can be empty

        numba.cuda.experimental.structref.register(MyStruct)

    """
    if struct_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")
    default_manager.register(struct_type, models.StructRefModel)
    define_attributes(struct_type)
    return struct_type


class StructRefProxy:
    """A PyObject proxy to the Numba allocated structref data structure.

    Notes
    -----

    * Subclasses should not define ``__init__``.
    * Subclasses can override ``__new__``.
    """

    __slots__ = ("_type", "_meminfo")

    @classmethod
    def _numba_box_(cls, ty, mi):
        """Called by boxing logic, the conversion of Numba internal
        representation into a PyObject.

        Parameters
        ----------
        ty :
            a Numba type instance.
        mi :
            a wrapped MemInfoPointer.

        Returns
        -------
        instance :
             a StructRefProxy instance.
        """
        instance = super().__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        return instance

    def __new__(cls, *args):
        """Construct a new instance of the structref.

        This takes positional-arguments only due to limitation of the compiler.
        The arguments are mapped to ``cls(*args)`` in jit-code.
        """
        try:
            # use cached ctor if available
            ctor = cls.__numba_ctor
        except AttributeError:
            # lazily define the ctor
            @jit
            def ctor(*args):
                return cls(*args)

            # cache it to attribute to avoid recompilation
            cls.__numba_ctor = ctor
        return ctor(*args)

    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        return self._type
