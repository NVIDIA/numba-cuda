from numba.core.lowering import Lower
from llvmlite import ir


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
            and (
                name not in self._singly_assigned_vars
                or self._disable_sroa_like_opt
            )
            # No emission of debuginfo for internal names
            and not name.startswith("$")
        ):
            # Emit debug value for user variable
            fetype = self.typeof(name)
            lltype = self.context.get_value_type(fetype)
            int_type = (ir.IntType,)
            real_type = ir.FloatType, ir.DoubleType
            if isinstance(lltype, int_type + real_type):
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
