from llvmlite import ir
from numba.core import (types, cgutils)
from numba.core.debuginfo import DIBuilder
from numba.cuda.types import GridGroup

_BYTE_SIZE = 8


class CUDADIBuilder(DIBuilder):
    def __init__(self, module, filepath, cgctx, directives_only):
        super().__init__(module, filepath, cgctx, directives_only)
        # Cache for local variable metadata type and line deduplication
        self._vartypelinemap = {}

    def _var_type(self, lltype, size, datamodel=None):
        is_bool = False
        is_int_literal = False
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
                    if isinstance(datamodel.fe_type, types.BooleanLiteral):
                        name = "bool"
                elif isinstance(datamodel.fe_type, types.Integer):
                    if isinstance(datamodel.fe_type, types.IntegerLiteral):
                        name = f"int{_BYTE_SIZE * size}"
                        is_int_literal = True
                elif isinstance(datamodel.fe_type, GridGroup):
                    is_grid_group = True

        if is_bool or is_int_literal or is_grid_group:
            m = self.module
            bitsize = _BYTE_SIZE * size
            # Boolean type workaround until upstream Numba is fixed
            if is_bool:
                ditok = "DW_ATE_boolean"
            elif is_int_literal:
                ditok = "DW_ATE_signed"
            # GridGroup type should use numba.cuda implementation
            elif is_grid_group:
                ditok = "DW_ATE_unsigned"

            return m.add_debug_info(
                "DIBasicType",
                {
                    "name": name,
                    "size": bitsize,
                    "encoding": ir.DIToken(ditok),
                },
            )

        # For other cases, use upstream Numba implementation
        return super()._var_type(lltype, size, datamodel=datamodel)

    def mark_variable(self, builder, allocavalue, name, lltype, size, line,
                      datamodel=None, argidx=None):
        if (name.startswith('$') or '.' in name):
            # Do not emit llvm.dbg.declare on user variable alias
            return
        else:
            int_type = ir.IntType,
            real_type = ir.FloatType, ir.DoubleType
            if (isinstance(lltype, int_type + real_type)):
                # Start with scalar variable, swtiching llvm.dbg.declare
                # to llvm.dbg.value
                return
            else:
                return super().mark_variable(builder, allocavalue, name, lltype,
                                             size, line, datamodel, argidx)

    def update_variable(self, builder, value, name, lltype, size, line,
                        datamodel=None, argidx=None):
        m = self.module
        fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        decl = cgutils.get_or_insert_function(m, fnty, 'llvm.dbg.value')

        mdtype = self._var_type(lltype, size, datamodel)
        index = name.find('.')
        if index >= 0:
            name = name[:index]
        # Merge DILocalVariable nodes with same name and type but different
        # lines. Use the cached [(name, type) -> line] info to deduplicate
        # metadata. Use the lltype as part of key.
        key = (name, lltype)
        if key in self._vartypelinemap:
            line = self._vartypelinemap[key]
        else:
            self._vartypelinemap[key] = line
        arg_index = 0 if argidx is None else argidx
        mdlocalvar = m.add_debug_info('DILocalVariable', {
            'name': name,
            'arg': arg_index,
            'scope': self.subprograms[-1],
            'file': self.difile,
            'line': line,
            'type': mdtype,
        })
        mdexpr = m.add_debug_info('DIExpression', {})

        return builder.call(decl, [value, mdlocalvar, mdexpr])
