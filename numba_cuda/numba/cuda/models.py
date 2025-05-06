import functools

from llvmlite import ir

from numba.core.datamodel.registry import DataModelManager, register
from numba.core.extending import models
from numba.core import types
from numba.cuda.types import Dim3, GridGroup, CUDADispatcher, CUDAArray
from numba.cuda.cudadrv import nvvm


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


@register_model(CUDAArray)
class CUDAArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type: CUDAArray):
        ndim = fe_type.ndim
        members = [
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            (
                "data",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super(CUDAArrayModel, self).__init__(dmm, fe_type, members)


@register_model(types.CPointer)
class GenericPointerModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        adrsp = (
            fe_type.addrspace
            if fe_type.addrspace is not None
            else nvvm.ADDRSPACE_GENERIC
        )
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)
