from numba.core.lowering import Lower
from llvmlite import ir
from numba.core import ir as numba_ir
from numba.core import types
from llvmlite.ir import Constant


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
                src_name = name.split(".")[0]
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
        self.poly_var_set = set()
        self.poly_cleaned = False
        self.lastblk = max(self.blocks.keys())

        # When debug info is enabled, walk through function body and mark
        # variables with polymorphic types.
        if self.context.enable_debuginfo and self._disable_sroa_like_opt:
            poly_map = {}
            # pre-scan all blocks
            for block in self.blocks.values():
                for x in block.find_insts(numba_ir.Assign):
                    if x.target.name.startswith("$"):
                        continue
                    ssa_name = x.target.name
                    src_name = ssa_name.split(".")[0]
                    # Check all the multi-versioned targets
                    if len(x.target.versioned_names) > 0:
                        fetype = self.typeof(ssa_name)
                        if src_name not in poly_map:
                            poly_map[src_name] = set()
                        # deduplicate polymorphic types
                        if isinstance(fetype, types.Literal):
                            fetype = fetype.literal_type
                        poly_map[src_name].add(fetype)
            # Filter out multi-versioned but single typed variables
            self.poly_var_typ_map = {
                k: v for k, v in poly_map.items() if len(v) > 1
            }

    def _alloca_var(self, name, fetype):
        """
        Ensure the given variable has an allocated stack slot (if needed).
        """
        # If the name is not handled yet and a store is needed
        if name not in self.varmap and self.store_var_needed(name):
            src_name = name.split(".")[0]
            if src_name in self.poly_var_typ_map:
                self.poly_var_set.add(name)
                if src_name not in self.poly_var_loc_map:
                    dtype = types.UnionType(self.poly_var_typ_map[src_name])
                    datamodel = self.context.data_model_manager[dtype]
                    # UnionType has sorted set of types, max at last index
                    maxsizetype = dtype.types[-1]
                    # Create a single element aggregate type
                    aggr_type = types.UniTuple(maxsizetype, 1)
                    lltype = self.context.get_value_type(aggr_type)
                    ptr = self.alloca_lltype(src_name, lltype, datamodel)
                    # save the location of the union type for polymorphic var
                    self.poly_var_loc_map[src_name] = ptr
                return

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

    def delvar(self, name):
        """
        Delete the given variable.
        """
        if name in self.poly_var_set:
            fetype = self.typeof(name)
            src_name = name.split(".")[0]
            ptr = self.poly_var_loc_map[src_name]
            self.decref(fetype, self.builder.load(ptr))
            if (
                self._cur_ir_block == self.blocks[self.lastblk]
                and not self.poly_cleaned
            ):
                # Zero-fill the debug union for polymorphic only
                # at the last block
                for v in self.poly_var_loc_map.values():
                    self.builder.store(Constant(v.type.pointee, None), v)
                    self.poly_cleaned = True
            return

        super().delvar(name)

    def getvar(self, name):
        """
        Get a pointer to the given variable's slot.
        """
        if name in self.poly_var_set:
            src_name = name.split(".")[0]
            fetype = self.typeof(name)
            lltype = self.context.get_value_type(fetype)
            castptr = self.builder.bitcast(
                self.poly_var_loc_map[src_name], ir.PointerType(lltype)
            )
            return castptr
        else:
            return super().getvar(name)
