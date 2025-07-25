import os

from llvmlite import ir
from numba.core import types, config
from numba.cuda import cgutils
from numba.core.datamodel.models import ComplexModel, UnionModel, UniTupleModel
from numba.core.debuginfo import AbstractDIBuilder
from numba.cuda.types import GridGroup

_BYTE_SIZE = 8


class DIBuilder(AbstractDIBuilder):
    DWARF_VERSION = 4
    DEBUG_INFO_VERSION = 3
    DBG_CU_NAME = "llvm.dbg.cu"
    _DEBUG = False

    def __init__(self, module, filepath, cgctx, directives_only):
        self.module = module
        self.filepath = os.path.abspath(filepath)
        self.difile = self._di_file()
        self.subprograms = []
        self.cgctx = cgctx

        if directives_only:
            self.emission_kind = "DebugDirectivesOnly"
        else:
            self.emission_kind = "FullDebug"

        self.initialize()

    def initialize(self):
        # Create the compile unit now because it is referenced when
        # constructing subprograms
        self.dicompileunit = self._di_compile_unit()

    def _var_type(self, lltype, size, datamodel=None):
        if self._DEBUG:
            print(
                "-->",
                lltype,
                size,
                datamodel,
                getattr(datamodel, "fe_type", "NO FE TYPE"),
            )
        m = self.module
        bitsize = _BYTE_SIZE * size

        int_type = (ir.IntType,)
        real_type = ir.FloatType, ir.DoubleType
        # For simple numeric types, choose the closest encoding.
        # We treat all integers as unsigned when there's no known datamodel.
        if isinstance(lltype, int_type + real_type):
            if datamodel is None:
                # This is probably something like an `i8*` member of a struct
                name = str(lltype)
                if isinstance(lltype, int_type):
                    ditok = "DW_ATE_unsigned"
                else:
                    ditok = "DW_ATE_float"
            else:
                # This is probably a known int/float scalar type
                name = str(datamodel.fe_type)
                if isinstance(datamodel.fe_type, types.Integer):
                    if datamodel.fe_type.signed:
                        ditok = "DW_ATE_signed"
                    else:
                        ditok = "DW_ATE_unsigned"
                else:
                    ditok = "DW_ATE_float"
            mdtype = m.add_debug_info(
                "DIBasicType",
                {
                    "name": name,
                    "size": bitsize,
                    "encoding": ir.DIToken(ditok),
                },
            )
        elif isinstance(datamodel, ComplexModel):
            # TODO: Is there a better way of determining "this is a complex
            # number"?
            #
            # NOTE: Commented below is the way to generate the metadata for a
            # C99 complex type that's directly supported by DWARF. Numba however
            # generates a struct with real/imag cf. CPython to give a more
            # pythonic feel to inspection.
            #
            # mdtype = m.add_debug_info('DIBasicType', {
            #  'name': f"{datamodel.fe_type} ({str(lltype)})",
            #  'size': bitsize,
            # 'encoding': ir.DIToken('DW_ATE_complex_float'),
            # })
            meta = []
            offset = 0
            for ix, name in enumerate(("real", "imag")):
                component = lltype.elements[ix]
                component_size = self.cgctx.get_abi_sizeof(component)
                component_basetype = m.add_debug_info(
                    "DIBasicType",
                    {
                        "name": str(component),
                        "size": _BYTE_SIZE * component_size,  # bits
                        "encoding": ir.DIToken("DW_ATE_float"),
                    },
                )
                derived_type = m.add_debug_info(
                    "DIDerivedType",
                    {
                        "tag": ir.DIToken("DW_TAG_member"),
                        "name": name,
                        "baseType": component_basetype,
                        "size": _BYTE_SIZE
                        * component_size,  # DW_TAG_member size is in bits
                        "offset": offset,
                    },
                )
                meta.append(derived_type)
                offset += _BYTE_SIZE * component_size  # offset is in bits
            mdtype = m.add_debug_info(
                "DICompositeType",
                {
                    "tag": ir.DIToken("DW_TAG_structure_type"),
                    "name": f"{datamodel.fe_type} ({str(lltype)})",
                    "identifier": str(lltype),
                    "elements": m.add_metadata(meta),
                    "size": offset,
                },
                is_distinct=True,
            )
        elif isinstance(datamodel, UniTupleModel):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            name = f"{datamodel.fe_type} ({str(lltype)})"
            count = size // el_size
            mdrange = m.add_debug_info(
                "DISubrange",
                {
                    "count": count,
                },
            )
            mdtype = m.add_debug_info(
                "DICompositeType",
                {
                    "tag": ir.DIToken("DW_TAG_array_type"),
                    "baseType": basetype,
                    "name": name,
                    "size": bitsize,
                    "identifier": str(lltype),
                    "elements": m.add_metadata([mdrange]),
                },
            )
        elif isinstance(lltype, ir.PointerType):
            model = getattr(datamodel, "_pointee_model", None)
            basetype = self._var_type(
                lltype.pointee, self.cgctx.get_abi_sizeof(lltype.pointee), model
            )
            mdtype = m.add_debug_info(
                "DIDerivedType",
                {
                    "tag": ir.DIToken("DW_TAG_pointer_type"),
                    "baseType": basetype,
                    "size": _BYTE_SIZE * self.cgctx.get_abi_sizeof(lltype),
                },
            )
        elif isinstance(lltype, ir.LiteralStructType):
            # Struct type
            meta = []
            offset = 0
            if datamodel is None or not datamodel.inner_models():
                name = f"Anonymous struct ({str(lltype)})"
                for field_id, element in enumerate(lltype.elements):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size)
                    derived_type = m.add_debug_info(
                        "DIDerivedType",
                        {
                            "tag": ir.DIToken("DW_TAG_member"),
                            "name": f"<field {field_id}>",
                            "baseType": basetype,
                            "size": _BYTE_SIZE
                            * size,  # DW_TAG_member size is in bits
                            "offset": offset,
                        },
                    )
                    meta.append(derived_type)
                    offset += _BYTE_SIZE * size  # offset is in bits
            else:
                name = f"{datamodel.fe_type} ({str(lltype)})"
                for element, field, model in zip(
                    lltype.elements, datamodel._fields, datamodel.inner_models()
                ):
                    size = self.cgctx.get_abi_sizeof(element)
                    basetype = self._var_type(element, size, datamodel=model)
                    derived_type = m.add_debug_info(
                        "DIDerivedType",
                        {
                            "tag": ir.DIToken("DW_TAG_member"),
                            "name": field,
                            "baseType": basetype,
                            "size": _BYTE_SIZE
                            * size,  # DW_TAG_member size is in bits
                            "offset": offset,
                        },
                    )
                    meta.append(derived_type)
                    offset += _BYTE_SIZE * size  # offset is in bits

            mdtype = m.add_debug_info(
                "DICompositeType",
                {
                    "tag": ir.DIToken("DW_TAG_structure_type"),
                    "name": name,
                    "identifier": str(lltype),
                    "elements": m.add_metadata(meta),
                    "size": offset,
                },
                is_distinct=True,
            )
        elif isinstance(lltype, ir.ArrayType):
            element = lltype.element
            el_size = self.cgctx.get_abi_sizeof(element)
            basetype = self._var_type(element, el_size)
            count = size // el_size
            mdrange = m.add_debug_info(
                "DISubrange",
                {
                    "count": count,
                },
            )
            mdtype = m.add_debug_info(
                "DICompositeType",
                {
                    "tag": ir.DIToken("DW_TAG_array_type"),
                    "baseType": basetype,
                    "name": str(lltype),
                    "size": bitsize,
                    "identifier": str(lltype),
                    "elements": m.add_metadata([mdrange]),
                },
            )
        else:
            # For all other types, describe it as sequence of bytes
            count = size
            mdrange = m.add_debug_info(
                "DISubrange",
                {
                    "count": count,
                },
            )
            mdbase = m.add_debug_info(
                "DIBasicType",
                {
                    "name": "byte",
                    "size": _BYTE_SIZE,
                    "encoding": ir.DIToken("DW_ATE_unsigned_char"),
                },
            )
            mdtype = m.add_debug_info(
                "DICompositeType",
                {
                    "tag": ir.DIToken("DW_TAG_array_type"),
                    "baseType": mdbase,
                    "name": str(lltype),
                    "size": bitsize,
                    "identifier": str(lltype),
                    "elements": m.add_metadata([mdrange]),
                },
            )

        return mdtype

    def mark_variable(
        self,
        builder,
        allocavalue,
        name,
        lltype,
        size,
        line,
        datamodel=None,
        argidx=None,
    ):
        arg_index = 0 if argidx is None else argidx
        m = self.module
        fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        decl = cgutils.get_or_insert_function(m, fnty, "llvm.dbg.declare")

        mdtype = self._var_type(lltype, size, datamodel=datamodel)
        name = name.replace(".", "$")  # for gdb to work correctly
        mdlocalvar = m.add_debug_info(
            "DILocalVariable",
            {
                "name": name,
                "arg": arg_index,
                "scope": self.subprograms[-1],
                "file": self.difile,
                "line": line,
                "type": mdtype,
            },
        )
        mdexpr = m.add_debug_info("DIExpression", {})

        return builder.call(decl, [allocavalue, mdlocalvar, mdexpr])

    def mark_location(self, builder, line):
        builder.debug_metadata = self._add_location(line)

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        name = qualname
        argmap = dict(zip(argnames, argtypes))
        di_subp = self._add_subprogram(
            name=name,
            linkagename=function.name,
            line=line,
            function=function,
            argmap=argmap,
        )
        function.set_metadata("dbg", di_subp)

    def finalize(self):
        dbgcu = cgutils.get_or_insert_named_metadata(
            self.module, self.DBG_CU_NAME
        )
        dbgcu.add(self.dicompileunit)
        self._set_module_flags()

    #
    # Internal APIs
    #

    def _set_module_flags(self):
        """Set the module flags metadata"""
        module = self.module
        mflags = cgutils.get_or_insert_named_metadata(
            module, "llvm.module.flags"
        )
        # Set *require* behavior to warning
        # See http://llvm.org/docs/LangRef.html#module-flags-metadata
        require_warning_behavior = self._const_int(2)
        if self.DWARF_VERSION is not None:
            dwarf_version = module.add_metadata(
                [
                    require_warning_behavior,
                    "Dwarf Version",
                    self._const_int(self.DWARF_VERSION),
                ]
            )
            if dwarf_version not in mflags.operands:
                mflags.add(dwarf_version)
        debuginfo_version = module.add_metadata(
            [
                require_warning_behavior,
                "Debug Info Version",
                self._const_int(self.DEBUG_INFO_VERSION),
            ]
        )
        if debuginfo_version not in mflags.operands:
            mflags.add(debuginfo_version)

    def _add_subprogram(self, name, linkagename, line, function, argmap):
        """Emit subprogram metadata"""
        subp = self._di_subprogram(name, linkagename, line, function, argmap)
        self.subprograms.append(subp)
        return subp

    def _add_location(self, line):
        """Emit location metatdaa"""
        loc = self._di_location(line)
        return loc

    @classmethod
    def _const_int(cls, num, bits=32):
        """Util to create constant int in metadata"""
        return ir.IntType(bits)(num)

    @classmethod
    def _const_bool(cls, boolean):
        """Util to create constant boolean in metadata"""
        return ir.IntType(1)(boolean)

    #
    # Helpers to emit the metadata nodes
    #

    def _di_file(self):
        return self.module.add_debug_info(
            "DIFile",
            {
                "directory": os.path.dirname(self.filepath),
                "filename": os.path.basename(self.filepath),
            },
        )

    def _di_compile_unit(self):
        return self.module.add_debug_info(
            "DICompileUnit",
            {
                "language": ir.DIToken("DW_LANG_C_plus_plus"),
                "file": self.difile,
                # Numba has to pretend to be clang to ensure the prologue is skipped
                # correctly in gdb. See:
                # https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/amd64-tdep.c;h=e563d369d8cb3eb3c2f732c2fa850ec70ba8d63b;hb=a4b0231e179607e47b1cdf1fe15c5dc25e482fad#l2521
                # Note the "producer_is_llvm" call to specialise the prologue
                # handling, this is defined here:
                # https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/producer.c;h=cdfd80d904c09394febd18749bb90359b2d128cc;hb=a4b0231e179607e47b1cdf1fe15c5dc25e482fad#l124
                # and to get a match for this condition the 'producer' must start
                # with "clang ", hence the following...
                "producer": "clang (Numba)",
                "runtimeVersion": 0,
                "isOptimized": config.OPT != 0,
                "emissionKind": ir.DIToken(self.emission_kind),
            },
            is_distinct=True,
        )

    def _di_subroutine_type(self, line, function, argmap):
        # The function call conv needs encoding.
        llfunc = function
        md = []

        for idx, llarg in enumerate(llfunc.args):
            if not llarg.name.startswith("arg."):
                name = llarg.name.replace(".", "$")  # for gdb to work correctly
                lltype = llarg.type
                size = self.cgctx.get_abi_sizeof(lltype)
                mdtype = self._var_type(lltype, size, datamodel=None)
                md.append(mdtype)

        for idx, (name, nbtype) in enumerate(argmap.items()):
            name = name.replace(".", "$")  # for gdb to work correctly
            datamodel = self.cgctx.data_model_manager[nbtype]
            lltype = self.cgctx.get_value_type(nbtype)
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=datamodel)
            md.append(mdtype)

        return self.module.add_debug_info(
            "DISubroutineType",
            {
                "types": self.module.add_metadata(md),
            },
        )

    def _di_subprogram(self, name, linkagename, line, function, argmap):
        return self.module.add_debug_info(
            "DISubprogram",
            {
                "name": name,
                "linkageName": linkagename,
                "scope": self.difile,
                "file": self.difile,
                "line": line,
                "type": self._di_subroutine_type(line, function, argmap),
                "isLocal": False,
                "isDefinition": True,
                "scopeLine": line,
                "isOptimized": config.OPT != 0,
                "unit": self.dicompileunit,
            },
            is_distinct=True,
        )

    def _di_location(self, line):
        return self.module.add_debug_info(
            "DILocation",
            {
                "line": line,
                "column": 1,
                "scope": self.subprograms[-1],
            },
        )


class CUDADIBuilder(DIBuilder):
    def __init__(self, module, filepath, cgctx, directives_only):
        super().__init__(module, filepath, cgctx, directives_only)
        # Cache for local variable metadata type and line deduplication
        self._vartypelinemap = {}

    def _var_type(self, lltype, size, datamodel=None):
        is_bool = False
        is_int_literal = False
        is_grid_group = False
        m = self.module

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

        if isinstance(datamodel, UnionModel):
            # UnionModel is handled here to represent polymorphic types
            meta = []
            maxwidth = 0
            for field, model in zip(
                datamodel._fields, datamodel.inner_models()
            ):
                # Ignore the "tag" field, focus on the "payload" field which
                # contains the data types in memory
                if field == "payload":
                    for mod in model.inner_models():
                        dtype = mod.get_value_type()
                        membersize = self.cgctx.get_abi_sizeof(dtype)
                        basetype = self._var_type(
                            dtype, membersize, datamodel=mod
                        )
                        if isinstance(mod.fe_type, types.Literal):
                            typename = str(mod.fe_type.literal_type)
                        else:
                            typename = str(mod.fe_type)
                        # Use a prefix "_" on type names as field names
                        membername = "_" + typename
                        memberwidth = _BYTE_SIZE * membersize
                        derived_type = m.add_debug_info(
                            "DIDerivedType",
                            {
                                "tag": ir.DIToken("DW_TAG_member"),
                                "name": membername,
                                "baseType": basetype,
                                # DW_TAG_member size is in bits
                                "size": memberwidth,
                            },
                        )
                        meta.append(derived_type)
                        if memberwidth > maxwidth:
                            maxwidth = memberwidth

            fake_union_name = "dbg_poly_union"
            return m.add_debug_info(
                "DICompositeType",
                {
                    "file": self.difile,
                    "tag": ir.DIToken("DW_TAG_union_type"),
                    "name": fake_union_name,
                    "identifier": str(lltype),
                    "elements": m.add_metadata(meta),
                    "size": maxwidth,
                },
                is_distinct=True,
            )
        # For other cases, use upstream Numba implementation
        return super()._var_type(lltype, size, datamodel=datamodel)

    def _di_subroutine_type(self, line, function, argmap):
        # The function call conv needs encoding.
        llfunc = function
        md = []

        # Create metadata type for return value
        if len(llfunc.args) > 0:
            lltype = llfunc.args[0].type
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=None)
            md.append(mdtype)

        # Create metadata type for arguments
        for idx, (name, nbtype) in enumerate(argmap.items()):
            datamodel = self.cgctx.data_model_manager[nbtype]
            lltype = self.cgctx.get_value_type(nbtype)
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=datamodel)
            md.append(mdtype)

        return self.module.add_debug_info(
            "DISubroutineType",
            {
                "types": self.module.add_metadata(md),
            },
        )

    def mark_variable(
        self,
        builder,
        allocavalue,
        name,
        lltype,
        size,
        line,
        datamodel=None,
        argidx=None,
    ):
        if name.startswith("$") or "." in name:
            # Do not emit llvm.dbg.declare on user variable alias
            return
        else:
            int_type = (ir.IntType,)
            real_type = ir.FloatType, ir.DoubleType
            if isinstance(lltype, int_type + real_type):
                # Start with scalar variable, swtiching llvm.dbg.declare
                # to llvm.dbg.value
                return
            else:
                return super().mark_variable(
                    builder,
                    allocavalue,
                    name,
                    lltype,
                    size,
                    line,
                    datamodel,
                    argidx,
                )

    def update_variable(
        self,
        builder,
        value,
        name,
        lltype,
        size,
        line,
        datamodel=None,
        argidx=None,
    ):
        m = self.module
        fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
        decl = cgutils.get_or_insert_function(m, fnty, "llvm.dbg.value")

        mdtype = self._var_type(lltype, size, datamodel)
        index = name.find(".")
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
        mdlocalvar = m.add_debug_info(
            "DILocalVariable",
            {
                "name": name,
                "arg": arg_index,
                "scope": self.subprograms[-1],
                "file": self.difile,
                "line": line,
                "type": mdtype,
            },
        )
        mdexpr = m.add_debug_info("DIExpression", {})

        return builder.call(decl, [value, mdlocalvar, mdexpr])
