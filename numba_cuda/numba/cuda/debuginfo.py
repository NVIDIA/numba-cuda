from llvmlite import ir
from numba.core import types
from numba.core.debuginfo import DIBuilder

_BYTE_SIZE = 8


class CUDADIBuilder(DIBuilder):

    def _var_type(self, lltype, size, datamodel=None):
        is_bool = False

        if isinstance(lltype, ir.IntType):
            if datamodel is None:
                if size == 1:
                    name = str(lltype)
                    is_bool = True
            elif isinstance(datamodel.fe_type, types.Boolean):
                name = str(datamodel.fe_type)
                is_bool = True

        # Booleans should use our implementation until upstream Numba is fixed
        if is_bool:
            m = self.module
            bitsize = _BYTE_SIZE * size
            ditok = "DW_ATE_boolean"

            return m.add_debug_info('DIBasicType', {
                'name': name,
                'size': bitsize,
                'encoding': ir.DIToken(ditok),
            })

        # For other cases, use upstream Numba implementation
        return super()._var_type(lltype, size, datamodel=datamodel)
