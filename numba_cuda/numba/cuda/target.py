# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
import warnings

from numba.core import types

from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaWarning
from numba.cuda.core.base import BaseContext
from numba.core.typing import cmathdecl
from numba.core import datamodel

from .cudadrv import nvvm
from numba.cuda import (
    cgutils,
    itanium_mangler,
    compiler,
    codegen,
    ufuncs,
    typing,
)
from numba.cuda.debuginfo import CUDADIBuilder
from numba.cuda.flags import CUDAFlags
from numba.cuda.models import cuda_data_manager
from numba.cuda.core.callconv import BaseCallConv, MinimalCallConv
from numba.cuda.core import config, targetconfig


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import (
            cudadecl,
            cudamath,
            fp16,
            bf16,
            libdevicedecl,
            vector_types,
        )
        from numba.cuda.typing import enumdecl, cffi_utils

        self.install_registry(cudadecl.registry)
        self.install_registry(cffi_utils.registry)
        self.install_registry(cudamath.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(libdevicedecl.registry)
        self.install_registry(enumdecl.registry)
        self.install_registry(vector_types.typing_registry)
        self.install_registry(fp16.typing_registry)
        self.install_registry(bf16.typing_registry)

    def resolve_value_type(self, val):
        # treat other dispatcher object as another device function
        from numba.cuda.dispatcher import CUDADispatcher

        if isinstance(val, Dispatcher) and not isinstance(val, CUDADispatcher):
            try:
                # use cached device function
                val = val.__dispatcher
            except AttributeError:
                if not val._can_compile:
                    raise ValueError(
                        "using cpu function on device "
                        "but its compilation is disabled"
                    )
                targetoptions = val.targetoptions.copy()
                targetoptions["device"] = True
                targetoptions["debug"] = targetoptions.get("debug", False)
                targetoptions["opt"] = targetoptions.get("opt", True)
                disp = CUDADispatcher(val.py_func, targetoptions)
                # cache the device function for future use and to avoid
                # duplicated copy of the same function.
                val.__dispatcher = disp
                val = disp

        # continue with parent logic
        return super(CUDATypingContext, self).resolve_value_type(val)

    def can_convert(self, fromty, toty):
        """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.typeconv.Conversion instance;
        otherwise None is returned.
        """

        # This implementation works around the issue addressed in Numba PR
        # #10047, "Fix IntEnumMember.can_convert_to() when no conversions
        # found", https://github.com/numba/numba/pull/10047.
        #
        # This should be gated on the version of Numba that the fix is
        # incorporated into, and eventually removed when the minimum supported
        # Numba version includes the fix.

        try:
            return super().can_convert(fromty, toty)
        except TypeError:
            if isinstance(fromty, types.IntEnumMember):
                # IntEnumMember fails to correctly handle impossible
                # conversions - in this scenario the correct thing to do is to
                # return None to signal that the conversion was not possible
                return None
            else:
                # Any failure involving conversion from a non-IntEnumMember is
                # almost certainly a real and separate issue
                raise


# -----------------------------------------------------------------------------
# Implementation


VALID_CHARS = re.compile(r"[^a-z0-9]", re.I)


class CUDATargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True

    def __init__(self, typingctx, target="cuda"):
        super().__init__(typingctx, target)
        self.data_model_manager = cuda_data_manager.chain(
            datamodel.default_manager
        )

    @property
    def enable_nrt(self):
        return getattr(config, "CUDA_ENABLE_NRT", False)

    @property
    def DIBuilder(self):
        return CUDADIBuilder

    @property
    def enable_boundscheck(self):
        # Unconditionally disabled
        return False

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self._internal_codegen = codegen.JITCUDACodegen("numba.cuda.jit")
        self._target_data = None

    def load_additional_registries(self):
        # side effect of import needed for numba.cpython.*, numba.cuda.cpython.*, the builtins
        # registry is updated at import time.
        from numba.cpython import tupleobj  # noqa: F401
        from numba.cuda.cpython import (
            numbers,
            slicing,
            iterators,
            listobj,
            unicode,
            charseq,
            cmathimpl,
            mathimpl,
        )
        from numba.cpython import rangeobj, enumimpl  # noqa: F401
        from numba.cuda.core import optional  # noqa: F401
        from numba.cuda.misc import cffiimpl
        from numba.cuda.np import (
            arrayobj,
            npdatetime,
            polynomial,
        )
        from . import (
            cudaimpl,
            fp16,
            printimpl,
            libdeviceimpl,
            mathimpl as cuda_mathimpl,
            vector_types,
            bf16,
        )

        # fix for #8940
        from numba.cuda.np.unsafe import ndarray  # noqa F401

        self.install_registry(cudaimpl.registry)
        self.install_registry(cffiimpl.registry)
        self.install_registry(printimpl.registry)
        self.install_registry(libdeviceimpl.registry)
        self.install_registry(cmathimpl.registry)
        self.install_registry(mathimpl.registry)
        self.install_registry(numbers.registry)
        self.install_registry(optional.registry)
        self.install_registry(cuda_mathimpl.registry)
        self.install_registry(vector_types.impl_registry)
        self.install_registry(fp16.target_registry)
        self.install_registry(bf16.target_registry)
        self.install_registry(slicing.registry)
        self.install_registry(iterators.registry)
        self.install_registry(listobj.registry)
        self.install_registry(unicode.registry)
        self.install_registry(charseq.registry)

        # install np registries
        self.install_registry(polynomial.registry)
        self.install_registry(npdatetime.registry)
        self.install_registry(arrayobj.registry)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        if self._target_data is None:
            self._target_data = ll.create_target_data(nvvm.NVVM().data_layout)
        return self._target_data

    @cached_property
    def nonconst_module_attrs(self):
        """
        Some CUDA intrinsics are at the module level, but cannot be treated as
        constants, because they are loaded from a special register in the PTX.
        These include threadIdx, blockDim, etc.
        """
        from numba import cuda

        nonconsts = (
            "threadIdx",
            "blockDim",
            "blockIdx",
            "gridDim",
            "laneid",
            "warpsize",
        )
        nonconsts_with_mod = tuple(
            [(types.Module(cuda), nc) for nc in nonconsts]
        )
        return nonconsts_with_mod

    @cached_property
    def call_conv(self):
        return CUDACallConv(self)

    def mangler(self, name, argtypes, *, abi_tags=(), uid=None):
        return itanium_mangler.mangle(
            name, argtypes, abi_tags=abi_tags, uid=uid
        )

    def make_constant_array(self, builder, aryty, arr):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """

        lmod = builder.module

        constvals = [
            self.get_constant(types.byte, i)
            for i in iter(arr.tobytes(order="A"))
        ]
        constaryty = ir.ArrayType(ir.IntType(8), len(constvals))
        constary = ir.Constant(constaryty, constvals)

        addrspace = nvvm.ADDRSPACE_CONSTANT
        gv = cgutils.add_global_variable(
            lmod, constary.type, "_cudapy_cmem", addrspace=addrspace
        )
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = constary

        # Preserve the underlying alignment
        lldtype = self.get_data_type(aryty.dtype)
        align = self.get_abi_sizeof(lldtype)
        gv.align = 2 ** (align - 1).bit_length()

        # Convert to generic address-space
        ptrty = ir.PointerType(ir.IntType(8))
        genptr = builder.addrspacecast(gv, ptrty, "generic")

        # Create array object
        ary = self.make_array(aryty)(self, builder)
        kshape = [self.get_constant(types.intp, s) for s in arr.shape]
        kstrides = [self.get_constant(types.intp, s) for s in arr.strides]
        self.populate_array(
            ary,
            data=builder.bitcast(genptr, ary.data.type),
            shape=kshape,
            strides=kstrides,
            itemsize=ary.itemsize,
            parent=ary.parent,
            meminfo=None,
        )

        return ary._getvalue()

    def insert_const_string(self, mod, string):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
        text = cgutils.make_bytearray(string.encode("utf-8") + b"\x00")
        name = "$".join(
            ["__conststring__", itanium_mangler.mangle_identifier(string)]
        )
        # Try to reuse existing global
        gv = mod.globals.get(name)
        if gv is None:
            # Not defined yet
            gv = cgutils.add_global_variable(
                mod, text.type, name, addrspace=nvvm.ADDRSPACE_CONSTANT
            )
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = text

        # Cast to a i8* pointer
        charty = gv.type.pointee.element
        return gv.bitcast(charty.as_pointer(nvvm.ADDRSPACE_CONSTANT))

    def insert_string_const_addrspace(self, builder, string):
        """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
        lmod = builder.module
        gv = self.insert_const_string(lmod, string)
        charptrty = ir.PointerType(ir.IntType(8))
        return builder.addrspacecast(gv, charptrty, "generic")

    def optimize_function(self, func):
        """Run O1 function passes"""
        pass
        ## XXX skipped for now
        # fpm = lp.FunctionPassManager.new(func.module)
        #
        # lp.PassManagerBuilder.new().populate(fpm)
        #
        # fpm.initialize()
        # fpm.run(func)
        # fpm.finalize()

    def get_ufunc_info(self, ufunc_key):
        return ufuncs.get_ufunc_info(ufunc_key)

    def _compile_subroutine_no_cache(
        self, builder, impl, sig, locals=None, flags=None
    ):
        # Overrides numba.core.base.BaseContext._compile_subroutine_no_cache().
        # Modified to use flags from the context stack if they are not provided
        # (pending a fix in Numba upstream).

        if locals is None:
            locals = {}

        with global_compiler_lock:
            codegen = self.codegen()
            library = codegen.create_library(impl.__name__)
            if flags is None:
                cstk = targetconfig.ConfigStack()
                if cstk:
                    flags = cstk.top().copy()
                else:
                    msg = "There should always be a context stack; none found."
                    warnings.warn(msg, NumbaWarning)
                    flags = CUDAFlags()

            flags.no_compile = True
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True

            cres = compiler.compile_internal(
                self.typing_context,
                self,
                library,
                impl,
                sig.args,
                sig.return_type,
                flags,
                locals=locals,
            )

            # Allow inlining the function inside callers
            self.active_code_library.add_linking_library(cres.library)
            return cres


class CUDACallConv(MinimalCallConv):
    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        # Do not prefix "arg." on argument name, so that nvvm compiler
        # can track debug info of argument more accurately
        arginfo.assign_names(self.get_arguments(fn), args)
        fn.args[0].name = ".ret"


class CUDACABICallConv(BaseCallConv):
    """
    Calling convention aimed at matching the CUDA C/C++ ABI. The implemented
    function signature is:

        <Python return type> (<Python arguments>)

    Exceptions are unsupported in this convention.
    """

    def _make_call_helper(self, builder):
        # Call helpers are used to help report exceptions back to Python, so
        # none is required here.
        return None

    def return_value(self, builder, retval):
        return builder.ret(retval)

    def return_user_exc(
        self, builder, exc, exc_args=None, loc=None, func_name=None
    ):
        msg = "Python exceptions are unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder, status):
        msg = "Return status is unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)

    def get_function_type(self, restype, argtypes):
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        fnty = ir.FunctionType(self.get_return_type(restype), argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn), ["arg." + a for a in args])

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        arginfo = self._get_arg_packer(argtys)
        realargs = arginfo.as_arguments(builder, args)
        code = builder.call(callee, realargs)
        # No status required as we don't support exceptions or a distinct None
        # value in a C ABI.
        status = None
        out = self.context.get_returned_value(builder, resty, code)
        return status, out

    def get_return_type(self, ty):
        return self.context.data_model_manager[ty].get_return_type()
