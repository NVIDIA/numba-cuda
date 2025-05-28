from numba.core.lowering import Lower
from llvmlite import ir
from numba.core import ir as numba_ir
from numba.core import types

class CUDALower(Lower):
    def storevar(self, value, name, argidx=None):
        """
        Store the value into the given variable.
        """
        super().storevar(value, name, argidx)

        # Emit llvm.dbg.value instead of llvm.dbg.declare for local scalar
        # variables immediately after a store instruction.
        if (
            self.context.enable_debuginfo
            # Conditions used to elide stores in parent method
            and self.store_var_needed(name)
            # No emission of debuginfo for internal names
            and not name.startswith("$")
        ):
            # Emit debug value for user variable
            fetype = self.typeof(name)
            lltype = self.context.get_value_type(fetype)
            int_type = (ir.IntType,)
            real_type = ir.FloatType, ir.DoubleType
            if isinstance(lltype, int_type + real_type):
                index = name.find(".")
                src_name = name[:index] if index > 0 else name
                if src_name in self.poly_var_typ_map:
                    # Do not emit debug value on polymorphic type var
                    return
                # Emit debug value for scalar variable
                sizeof = self.context.get_abi_sizeof(lltype)
                datamodel = self.context.data_model_manager[fetype]
                line = self.loc.line if argidx is None else self.defn_loc.line
                self.debuginfo.update_variable(
                    self.builder,
                    value,
                    name,
                    lltype,
                    sizeof,
                    line,
                    datamodel,
                    argidx,
                )

    def pre_lower(self):
        """
        Called before lowering all blocks.
        """
        super().pre_lower()

        self.poly_var_typ_map = {}
        self.poly_var_loc_map = {}

        # When debug info is enabled, walk through function body and mark
        # variables with polymorphic types.
        if self.context.enable_debuginfo and self._disable_sroa_like_opt:
            # pre-scan all blocks
            for block in self.blocks.values():
                for x in block.find_insts(numba_ir.Assign):
                    if x.target.name.startswith("$"):
                        continue
                    ssa_name = x.target.name
                    index = ssa_name.find(".")
                    src_name = ssa_name[:index] if index > 0 else ssa_name
                    if len(x.target.versioned_names) > 0:
                        fetype = self.typeof(ssa_name)
                        if src_name not in self.poly_var_typ_map:
                            self.poly_var_typ_map[src_name] = set()
                        # deduplicate polymorphic types
                        if isinstance(fetype, types.Literal):
                            fetype = fetype.literal_type
                        self.poly_var_typ_map[src_name].add(fetype)

    def _alloca_var(self, name, fetype):
        """
        Ensure the given variable has an allocated stack slot (if needed).
        """
        # If the name is not handled yet and a store is needed
        if (name not in self.varmap and self.store_var_needed(name)):
            index = name.find(".")
            src_name = name[:index] if index > 0 else name
            if src_name in self.poly_var_typ_map:
                dtype = types.UnionType(self.poly_var_typ_map[src_name])
                datamodel = self.context.data_model_manager[dtype]
                if src_name not in self.poly_var_loc_map:
                    # UnionType has sorted set of types, max at last index
                    maxsizetype = dtype.types[-1]
                    # Create a single element aggregate type
                    aggr_type = types.UniTuple(maxsizetype, 1)
                    lltype = self.context.get_value_type(aggr_type)
                    ptr = self.alloca_lltype(src_name, lltype, datamodel)
                    # save the location of the union type for polymorphic var
                    self.poly_var_loc_map[src_name] = ptr
                # Any member of this union type shoud type cast ptr to fetype
                lltype = self.context.get_value_type(fetype)
                castptr = self.builder.bitcast(self.poly_var_loc_map[src_name],
                                               ir.PointerType(lltype))
                # Remember the pointer
                self.varmap[name] = castptr

        super()._alloca_var(name, fetype)

    def store_var_needed(self, name):
        # Check the conditions used to elide stores in parent class,
        # e.g. in method storevar() and _alloca_var()
        return (
            # used in multiple blocks
            name not in self._singly_assigned_vars
            # lowering with debuginfo
            or self._disable_sroa_like_opt
        )
