from llvmlite import ir
from numba.core import types
from numba.core.debuginfo import DIBuilder
from numba.cuda.types import GridGroup

_BYTE_SIZE = 8


class CUDADIBuilder(DIBuilder):

    def _var_type(self, lltype, size, datamodel=None):
        is_bool = False
        is_grid_group = False

        if isinstance(lltype, ir.IntType):
            if datamodel is None:
                if size == 1:
                    name = str(lltype)
                    is_bool = True
            else:
                name = str(datamodel.fe_type)
                if isinstance(datamodel.fe_type, types.Boolean):
                    is_bool = True
                elif isinstance(datamodel.fe_type, GridGroup):
                    is_grid_group = True

        if is_bool or is_grid_group:
            m = self.module
            bitsize = _BYTE_SIZE * size
            # Boolean type workaround until upstream Numba is fixed
            if is_bool:
                ditok = "DW_ATE_boolean"
            # GridGroup type should use numba.cuda implementation
            elif is_grid_group:
                ditok = "DW_ATE_unsigned"

            return m.add_debug_info('DIBasicType', {
                'name': name,
                'size': bitsize,
                'encoding': ir.DIToken(ditok),
            })

        # For other cases, use upstream Numba implementation
        return super()._var_type(lltype, size, datamodel=datamodel)
