# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools

from llvmlite import ir

from numba.cuda.datamodel.registry import DataModelManager, register
from numba.cuda.datamodel import PrimitiveModel
from numba.cuda.datamodel.models import DataModel, StructModel
from numba.cuda.extending import core_models as models
from numba.cuda import types
from numba.cuda.types.ext_types import Dim3, GridGroup, CUDADispatcher, Bfloat16


cuda_data_manager = DataModelManager()

register_model = functools.partial(register, cuda_data_manager)


@register_model(Dim3)
class Dim3Model(StructModel):
    def __init__(self, dmm, fe_type):
        members = [("x", types.int32), ("y", types.int32), ("z", types.int32)]
        super().__init__(dmm, fe_type, members)


@register_model(GridGroup)
class GridGroupModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(64)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Float)
class FloatModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        if fe_type == types.float16:
            be_type = ir.IntType(16)
        elif fe_type == types.float32:
            be_type = ir.FloatType()
        elif fe_type == types.float64:
            be_type = ir.DoubleType()
        else:
            raise NotImplementedError(fe_type)
        super().__init__(dmm, fe_type, be_type)


register_model(CUDADispatcher)(models.OpaqueModel)


@register_model(types.NoneType)
class NoneTypeModel(DataModel):
    """Data model for ``types.NoneType`` (``types.void``).

    Shadows the ``OpaqueModel`` registration for ``types.NoneType`` that
    exists in ``default_manager`` (from upstream numba).  Because
    ``cuda_data_manager`` is the first map in the ``ChainMap`` built by
    ``CUDATargetContext.__init__``, this model takes priority without
    mutating the underlying ``default_manager``.

    This model intentionally returns *different* LLVM types from
    ``get_value_type()`` and ``get_return_type()`` because LLVM treats
    ``void`` as a function-return-only concept, not a first-class value:

    ``get_value_type()`` → ``i8*`` (opaque pointer)
        Used whenever a concrete LLVM value is needed: variable
        assignment, alloca, store/load, constants, boxing/unboxing of
        ``None``, and the Numba-ABI return-slot pointer.  LLVM forbids
        creating constants, pointers, or stack slots of ``void``, so an
        opaque ``i8*`` null serves as the runtime stand-in for ``None``.

    ``get_return_type()`` → ``ir.VoidType()``
        Used exclusively when building ``ir.FunctionType`` for a
        function's return signature.  Returning ``void`` here lets the
        C-ABI calling convention emit ``void foo(...)`` instead of the
        incorrect ``i8* foo(...)``, which fixes the ABI / LTO mismatch
        described in GitHub issue #845.
    """

    _ptr_type = ir.IntType(8).as_pointer()

    def get_value_type(self):
        return self._ptr_type

    def get_return_type(self):
        return ir.VoidType()

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


@register_model(Bfloat16)
class _model___nv_bfloat16(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(16)
        super().__init__(dmm, fe_type, be_type)
