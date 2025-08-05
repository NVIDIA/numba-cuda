import functools

from llvmlite import ir

from numba.core.datamodel.registry import DataModelManager, register
from numba.core.datamodel import PrimitiveModel
from numba.core.extending import models
from numba.core import types
from numba.cuda.types import Dim3, GridGroup, CUDADispatcher, Bfloat16


cuda_data_manager = DataModelManager()

register_model = functools.partial(register, cuda_data_manager)


@register_model(Dim3)
class Dim3Model(models.StructModel):
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
        super(FloatModel, self).__init__(dmm, fe_type, be_type)


register_model(CUDADispatcher)(models.OpaqueModel)


def _as_bfloat(value):
    # Step 1: Convert to float
    f = ir.types._as_float(value)
    # Step 2: Truncate (or round, we choose truncate) last 16 bits
    bf = f >> 16
    return bf


class BfloatType(ir.types._BaseFloatType):
    """Brain-float type"""

    null = "0.0"
    intrinsic_name = "bfloat"

    def __str__(self):
        return "bfloat"

    def format_constant(self, value):
        return ir.types._format_double(_as_bfloat(value))


BfloatType._create_instance()


@register_model(Bfloat16)
class _model___nv_bfloat16(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # be_type = BfloatType()
        be_type = ir.IntType(16)
        super(_model___nv_bfloat16, self).__init__(dmm, fe_type, be_type)
