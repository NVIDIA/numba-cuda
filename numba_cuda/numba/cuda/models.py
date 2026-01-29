# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools

from llvmlite import ir

from numba.cuda.datamodel.registry import DataModelManager, register
from numba.cuda.datamodel import PrimitiveModel
from numba.cuda.datamodel.models import StructModel
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


@register_model(Bfloat16)
class _model___nv_bfloat16(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(16)
        super().__init__(dmm, fe_type, be_type)
