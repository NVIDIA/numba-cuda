# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import os
import sys
import ctypes
import collections
import functools
import types as pytypes
import weakref
import uuid
import re
from warnings import warn

from numba.core import types, config, errors, entrypoints
from numba.cuda import serialize, utils
from numba import cuda

from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.cuda.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof

from numba.cuda import typing
from numba.cuda import types as cuda_types
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.core.bytecode import get_code_object
from numba.cuda.compiler import (
    compile_cuda,
    CUDACompiler,
    kernel_fixup,
    compile_extra,
)
from numba.cuda.core import sigutils
from numba.cuda.flags import Flags
from numba.cuda.cudadrv import driver, nvvm
from numba.cuda.locks import module_init_lock
from numba.cuda.core.caching import Cache, CacheImpl, NullCache
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (
    missing_launch_config_msg,
    normalize_kernel_dimensions,
)
from numba.cuda.cudadrv.linkable_code import LinkableCode
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.memory_management.nrt import rtsys, NRT_LIBRARY

from numba.cuda.cext import _dispatcher


cuda_fp16_math_funcs = [
    "hsin",
    "hcos",
    "hlog",
    "hlog10",
    "hlog2",
    "hexp",
    "hexp10",
    "hexp2",
    "hsqrt",
    "hrsqrt",
    "hfloor",
    "hceil",
    "hrcp",
    "hrint",
    "htrunc",
    "hdiv",
]

reshape_funcs = ["nocopy_empty_reshape", "numba_attempt_nocopy_reshape"]


class _Kernel(serialize.ReduceMixin):
    """
    CUDA Kernel specialized for a given set of argument types. When called, this
    object launches the kernel on the device.
    """

    NRT_functions = [
        "NRT_Allocate",
        "NRT_MemInfo_init",
        "NRT_MemInfo_new",
        "NRT_Free",
        "NRT_dealloc",
        "NRT_MemInfo_destroy",
        "NRT_MemInfo_call_dtor",
        "NRT_MemInfo_data_fast",
        "NRT_MemInfo_alloc_aligned",
        "NRT_Allocate_External",
        "NRT_decref",
        "NRT_incref",
    ]

    @global_compiler_lock
    def __init__(
        self,
        py_func,
        argtypes,
        link=None,
        debug=False,
        lineinfo=False,
        inline=False,
        forceinline=False,
        fastmath=False,
        extensions=None,
        max_registers=None,
        lto=False,
        opt=True,
        device=False,
        launch_bounds=None,
    ):
        if device:
            raise RuntimeError("Cannot compile a device function as a kernel")

        super().__init__()

        # _DispatcherBase.nopython_signatures() expects this attribute to be
        # present, because it assumes an overload is a CompileResult. In the
        # CUDA target, _Kernel instances are stored instead, so we provide this
        # attribute here to avoid duplicating nopython_signatures() in the CUDA
        # target with slight modifications.
        self.objectmode = False

        # The finalizer constructed by _DispatcherBase._make_finalizer also
        # expects overloads to be a CompileResult. It uses the entry_point to
        # remove a CompileResult from a target context. However, since we never
        # insert kernels into a target context (there is no need because they
        # cannot be called by other functions, only through the dispatcher) it
        # suffices to pretend we have an entry point of None.
        self.entry_point = None

        self.py_func = py_func
        self.argtypes = argtypes
        self.debug = debug
        self.lineinfo = lineinfo
        self.extensions = extensions or []
        self.launch_bounds = launch_bounds

        nvvm_options = {"fastmath": fastmath, "opt": 3 if opt else 0}

        if debug:
            nvvm_options["g"] = None

        cc = get_current_device().compute_capability

        cres = compile_cuda(
            self.py_func,
            types.void,
            self.argtypes,
            debug=self.debug,
            lineinfo=lineinfo,
            forceinline=forceinline,
            fastmath=fastmath,
            nvvm_options=nvvm_options,
            cc=cc,
            max_registers=max_registers,
            lto=lto,
        )
        tgt_ctx = cres.target_context
        lib = cres.library
        kernel = lib.get_function(cres.fndesc.llvm_func_name)
        lib._entry_name = cres.fndesc.llvm_func_name
        kernel_fixup(kernel, self.debug)
        nvvm.set_launch_bounds(kernel, launch_bounds)

        if not link:
            link = []

        asm = lib.get_asm_str()

        # The code library contains functions that require cooperative launch.
        self.cooperative = lib.use_cooperative
        # We need to link against cudadevrt if grid sync is being used.
        if self.cooperative:
            lib.needs_cudadevrt = True

        def link_to_library_functions(
            library_functions, library_path, prefix=None
        ):
            """
            Dynamically links to library functions by searching for their names
            in the specified library and linking to the corresponding source
            file.
            """
            if prefix is not None:
                library_functions = [
                    f"{prefix}{fn}" for fn in library_functions
                ]

            found_functions = [fn for fn in library_functions if f"{fn}" in asm]

            if found_functions:
                basedir = os.path.dirname(os.path.abspath(__file__))
                source_file_path = os.path.join(basedir, library_path)
                link.append(source_file_path)

            return found_functions

        # Link to the helper library functions if needed
        link_to_library_functions(reshape_funcs, "reshape_funcs.cu")

        self.maybe_link_nrt(link, tgt_ctx, asm)

        for filepath in link:
            lib.add_linking_file(filepath)

        # populate members
        self.entry_name = kernel.name
        self.signature = cres.signature
        self._type_annotation = cres.type_annotation
        self._codelibrary = lib
        self.call_helper = cres.call_helper

        # The following are referred to by the cache implementation. Note:
        # - There are no referenced environments in CUDA.
        # - Kernels don't have lifted code.
        self.target_context = tgt_ctx
        self.fndesc = cres.fndesc
        self.environment = cres.environment
        self._referenced_environments = []
        self.lifted = []

    def maybe_link_nrt(self, link, tgt_ctx, asm):
        """
        Add the NRT source code to the link if the neccesary conditions are met.
        NRT must be enabled for the CUDATargetContext, and either NRT functions
        must be detected in the kernel asm or an NRT enabled LinkableCode object
        must be passed.
        """

        if not tgt_ctx.enable_nrt:
            return

        all_nrt = "|".join(self.NRT_functions)
        pattern = (
            r"\.extern\s+\.func\s+(?:\s*\(.+\)\s*)?("
            + all_nrt
            + r")\s*\([^)]*\)\s*;"
        )
        link_nrt = False
        nrt_in_asm = re.findall(pattern, asm)
        if len(nrt_in_asm) > 0:
            link_nrt = True
        if not link_nrt:
            for file in link:
                if isinstance(file, LinkableCode):
                    if file.nrt:
                        link_nrt = True
                        break

        if link_nrt:
            link.append(NRT_LIBRARY)

    @property
    def library(self):
        return self._codelibrary

    @property
    def type_annotation(self):
        return self._type_annotation

    def _find_referenced_environments(self):
        return self._referenced_environments

    @property
    def codegen(self):
        return self.target_context.codegen()

    @property
    def argument_types(self):
        return tuple(self.signature.args)

    @classmethod
    def _rebuild(
        cls,
        cooperative,
        name,
        signature,
        codelibrary,
        debug,
        lineinfo,
        call_helper,
        extensions,
    ):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        # invoke parent constructor
        super(cls, instance).__init__()
        # populate members
        instance.entry_point = None
        instance.cooperative = cooperative
        instance.entry_name = name
        instance.signature = signature
        instance._type_annotation = None
        instance._codelibrary = codelibrary
        instance.debug = debug
        instance.lineinfo = lineinfo
        instance.call_helper = call_helper
        instance.extensions = extensions
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
        return dict(
            cooperative=self.cooperative,
            name=self.entry_name,
            signature=self.signature,
            codelibrary=self._codelibrary,
            debug=self.debug,
            lineinfo=self.lineinfo,
            call_helper=self.call_helper,
            extensions=self.extensions,
        )

    @module_init_lock
    def initialize_once(self, mod):
        if not mod.initialized:
            mod.setup()

    def bind(self):
        """
        Force binding to current CUDA context
        """
        cufunc = self._codelibrary.get_cufunc()

        self.initialize_once(cufunc.module)

        if (
            hasattr(self, "target_context")
            and self.target_context.enable_nrt
            and config.CUDA_NRT_STATS
        ):
            rtsys.ensure_initialized()
            rtsys.set_memsys_to_module(cufunc.module)
            # We don't know which stream the kernel will be launched on, so
            # we force synchronize here.
            cuda.synchronize()

    @property
    def regs_per_thread(self):
        """
        The number of registers used by each thread for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.regs

    @property
    def const_mem_size(self):
        """
        The amount of constant memory used by this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.const

    @property
    def shared_mem_per_block(self):
        """
        The amount of shared memory used per block for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.shared

    @property
    def max_threads_per_block(self):
        """
        The maximum allowable threads per block.
        """
        return self._codelibrary.get_cufunc().attrs.maxthreads

    @property
    def local_mem_per_thread(self):
        """
        The amount of local memory used per thread for this kernel.
        """
        return self._codelibrary.get_cufunc().attrs.local

    def inspect_llvm(self):
        """
        Returns the LLVM IR for this kernel.
        """
        return self._codelibrary.get_llvm_str()

    def inspect_asm(self, cc):
        """
        Returns the PTX code for this kernel.
        """
        return self._codelibrary.get_asm_str(cc=cc)

    def inspect_lto_ptx(self, cc):
        """
        Returns the PTX code for the external functions linked to this kernel.
        """
        return self._codelibrary.get_lto_ptx(cc=cc)

    def inspect_sass_cfg(self):
        """
        Returns the CFG of the SASS for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
        return self._codelibrary.get_sass_cfg()

    def inspect_sass(self):
        """
        Returns the SASS code for this kernel.

        Requires nvdisasm to be available on the PATH.
        """
        return self._codelibrary.get_sass()

    def inspect_types(self, file=None):
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
        if self._type_annotation is None:
            raise ValueError("Type annotation is not available")

        if file is None:
            file = sys.stdout

        print("%s %s" % (self.entry_name, self.argument_types), file=file)
        print("-" * 80, file=file)
        print(self._type_annotation, file=file)
        print("=" * 80, file=file)

    def max_cooperative_grid_blocks(self, blockdim, dynsmemsize=0):
        """
        Calculates the maximum number of blocks that can be launched for this
        kernel in a cooperative grid in the current context, for the given block
        and dynamic shared memory sizes.

        :param blockdim: Block dimensions, either as a scalar for a 1D block, or
                         a tuple for 2D or 3D blocks.
        :param dynsmemsize: Dynamic shared memory size in bytes.
        :return: The maximum number of blocks in the grid.
        """
        ctx = get_context()
        cufunc = self._codelibrary.get_cufunc()

        if isinstance(blockdim, tuple):
            blockdim = functools.reduce(lambda x, y: x * y, blockdim)
        active_per_sm = ctx.get_active_blocks_per_multiprocessor(
            cufunc, blockdim, dynsmemsize
        )
        sm_count = ctx.device.MULTIPROCESSOR_COUNT
        return active_per_sm * sm_count

    def launch(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # Prepare kernel
        cufunc = self._codelibrary.get_cufunc()

        if self.debug:
            excname = cufunc.name + "__errcode__"
            excmem, excsz = cufunc.module.get_global_symbol(excname)
            assert excsz == ctypes.sizeof(ctypes.c_int)
            excval = ctypes.c_int()
            excmem.memset(0, stream=stream)

        # Prepare arguments
        retr = []  # hold functors for writeback

        kernelargs = []
        for t, v in zip(self.argument_types, args):
            self._prepare_args(t, v, stream, retr, kernelargs)

        if driver.USE_NV_BINDING:
            stream_handle = stream and stream.handle.value or 0
        else:
            zero_stream = None
            stream_handle = stream and stream.handle or zero_stream

        # Invoke kernel
        driver.launch_kernel(
            cufunc.handle,
            *griddim,
            *blockdim,
            sharedmem,
            stream_handle,
            kernelargs,
            cooperative=self.cooperative,
        )

        if self.debug:
            driver.device_to_host(ctypes.addressof(excval), excmem, excsz)
            if excval.value != 0:
                # An error occurred
                def load_symbol(name):
                    mem, sz = cufunc.module.get_global_symbol(
                        "%s__%s__" % (cufunc.name, name)
                    )
                    val = ctypes.c_int()
                    driver.device_to_host(ctypes.addressof(val), mem, sz)
                    return val.value

                tid = [load_symbol("tid" + i) for i in "zyx"]
                ctaid = [load_symbol("ctaid" + i) for i in "zyx"]
                code = excval.value
                exccls, exc_args, loc = self.call_helper.get_exception(code)
                # Prefix the exception message with the source location
                if loc is None:
                    locinfo = ""
                else:
                    sym, filepath, lineno = loc
                    filepath = os.path.abspath(filepath)
                    locinfo = "In function %r, file %s, line %s, " % (
                        sym,
                        filepath,
                        lineno,
                    )
                # Prefix the exception message with the thread position
                prefix = "%stid=%s ctaid=%s" % (locinfo, tid, ctaid)
                if exc_args:
                    exc_args = ("%s: %s" % (prefix, exc_args[0]),) + exc_args[
                        1:
                    ]
                else:
                    exc_args = (prefix,)
                raise exccls(*exc_args)

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """

        # map the arguments using any extension you've registered
        for extension in reversed(self.extensions):
            ty, val = extension.prepare_args(ty, val, stream=stream, retr=retr)

        if isinstance(ty, types.Array):
            devary = wrap_arg(val).to_device(retr, stream)
            c_intp = ctypes.c_ssize_t

            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)

            ptr = driver.device_pointer(devary)

            if driver.USE_NV_BINDING:
                ptr = int(ptr)

            data = ctypes.c_void_p(ptr)

            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))

        elif isinstance(ty, types.CPointer):
            # Pointer arguments should be a pointer-sized integer
            kernelargs.append(ctypes.c_uint64(val))

        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, "c_%s" % ty)(val)
            kernelargs.append(cval)

        elif ty == types.float16:
            cval = ctypes.c_uint16(np.float16(val).view(np.uint16))
            kernelargs.append(cval)

        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)

        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)

        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)

        elif ty == types.complex64:
            kernelargs.append(ctypes.c_float(val.real))
            kernelargs.append(ctypes.c_float(val.imag))

        elif ty == types.complex128:
            kernelargs.append(ctypes.c_double(val.real))
            kernelargs.append(ctypes.c_double(val.imag))

        elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
            kernelargs.append(ctypes.c_int64(val.view(np.int64)))

        elif isinstance(ty, types.Record):
            devrec = wrap_arg(val).to_device(retr, stream)
            ptr = devrec.device_ctypes_pointer
            kernelargs.append(ptr)

        elif isinstance(ty, types.BaseTuple):
            assert len(ty) == len(val)
            for t, v in zip(ty, val):
                self._prepare_args(t, v, stream, retr, kernelargs)

        elif isinstance(ty, types.EnumMember):
            try:
                self._prepare_args(
                    ty.dtype, val.value, stream, retr, kernelargs
                )
            except NotImplementedError:
                raise NotImplementedError(ty, val)

        else:
            raise NotImplementedError(ty, val)


class ForAll(object):
    def __init__(self, dispatcher, ntasks, tpb, stream, sharedmem):
        if ntasks < 0:
            raise ValueError(
                "Can't create ForAll with negative task count: %s" % ntasks
            )
        self.dispatcher = dispatcher
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if self.ntasks == 0:
            return

        if self.dispatcher.specialized:
            specialized = self.dispatcher
        else:
            specialized = self.dispatcher.specialize(*args)
        blockdim = self._compute_thread_per_block(specialized)
        griddim = (self.ntasks + blockdim - 1) // blockdim

        return specialized[griddim, blockdim, self.stream, self.sharedmem](
            *args
        )

    def _compute_thread_per_block(self, dispatcher):
        tpb = self.thread_per_block
        # Prefer user-specified config
        if tpb != 0:
            return tpb
        # Else, ask the driver to give a good config
        else:
            ctx = get_context()
            # Dispatcher is specialized, so there's only one definition - get
            # it so we can get the cufunc from the code library
            kernel = next(iter(dispatcher.overloads.values()))
            kwargs = dict(
                func=kernel._codelibrary.get_cufunc(),
                b2d_func=0,  # dynamic-shared memory is constant to blksz
                memsize=self.sharedmem,
                blocksizelimit=1024,
            )
            _, tpb = ctx.get_max_potential_block_size(**kwargs)
            return tpb


class _LaunchConfiguration:
    def __init__(self, dispatcher, griddim, blockdim, stream, sharedmem):
        self.dispatcher = dispatcher
        self.griddim = griddim
        self.blockdim = blockdim
        self.stream = stream
        self.sharedmem = sharedmem

        if (
            config.CUDA_LOW_OCCUPANCY_WARNINGS
            and not config.DISABLE_PERFORMANCE_WARNINGS
        ):
            # Warn when the grid has fewer than 128 blocks. This number is
            # chosen somewhat heuristically - ideally the minimum is 2 times
            # the number of SMs, but the number of SMs varies between devices -
            # some very small GPUs might only have 4 SMs, but an H100-SXM5 has
            # 132. In general kernels should be launched with large grids
            # (hundreds or thousands of blocks), so warning when fewer than 128
            # blocks are used will likely catch most beginner errors, where the
            # grid tends to be very small (single-digit or low tens of blocks).
            min_grid_size = 128
            grid_size = griddim[0] * griddim[1] * griddim[2]
            if grid_size < min_grid_size:
                msg = (
                    f"Grid size {grid_size} will likely result in GPU "
                    "under-utilization due to low occupancy."
                )
                warn(errors.NumbaPerformanceWarning(msg))

    def __call__(self, *args):
        return self.dispatcher.call(
            args, self.griddim, self.blockdim, self.stream, self.sharedmem
        )


class CUDACacheImpl(CacheImpl):
    def reduce(self, kernel):
        return kernel._reduce_states()

    def rebuild(self, target_context, payload):
        return _Kernel._rebuild(**payload)

    def check_cachable(self, cres):
        # CUDA Kernels are always cachable - the reasons for an entity not to
        # be cachable are:
        #
        # - The presence of lifted loops, or
        # - The presence of dynamic globals.
        #
        # neither of which apply to CUDA kernels.
        return True


class CUDACache(Cache):
    """
    Implements a cache that saves and loads CUDA kernels and compile results.
    """

    _impl_class = CUDACacheImpl

    def load_overload(self, sig, target_context):
        # Loading an overload refreshes the context to ensure it is
        # initialized. To initialize the correct (i.e. CUDA) target, we need to
        # enforce that the current target is the CUDA target.
        from numba.core.target_extension import target_override

        with target_override("cuda"):
            return super().load_overload(sig, target_context)


class OmittedArg(object):
    """
    A placeholder for omitted arguments with a default value.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "omitted arg(%r)" % (self.value,)

    @property
    def _numba_type_(self):
        return types.Omitted(self.value)


class CompilingCounter(object):
    """
    A simple counter that increment in __enter__ and decrement in __exit__.
    """

    def __init__(self):
        self.counter = 0

    def __enter__(self):
        assert self.counter >= 0
        self.counter += 1

    def __exit__(self, *args, **kwargs):
        self.counter -= 1
        assert self.counter >= 0

    def __bool__(self):
        return self.counter > 0

    __nonzero__ = __bool__


class _DispatcherBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """

    __numba__ = "py_func"

    def __init__(
        self, arg_count, py_func, pysig, can_fallback, exact_match_required
    ):
        self._tm = default_type_manager

        # A mapping of signatures to compile results
        self.overloads = collections.OrderedDict()

        self.py_func = py_func
        # other parts of Numba assume the old Python 2 name for code object
        self.func_code = get_code_object(py_func)
        # but newer python uses a different name
        self.__code__ = self.func_code
        # a place to keep an active reference to the types of the active call
        self._types_active_call = set()
        # Default argument values match the py_func
        self.__defaults__ = py_func.__defaults__

        argnames = tuple(pysig.parameters)
        default_values = self.py_func.__defaults__ or ()
        defargs = tuple(OmittedArg(val) for val in default_values)
        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL
        _dispatcher.Dispatcher.__init__(
            self,
            self._tm.get_pointer(),
            arg_count,
            self._fold_args,
            argnames,
            defargs,
            can_fallback,
            has_stararg,
            exact_match_required,
        )

        self.doc = py_func.__doc__
        self._compiling_counter = CompilingCounter()
        weakref.finalize(self, self._make_finalizer())

    def _compilation_chain_init_hook(self):
        """
        This will be called ahead of any part of compilation taking place (this
        even includes being ahead of working out the types of the arguments).
        This permits activities such as initialising extension entry points so
        that the compiler knows about additional externally defined types etc
        before it does anything.
        """
        entrypoints.init_all()

    def _reset_overloads(self):
        self._clear()
        self.overloads.clear()

    def _make_finalizer(self):
        """
        Return a finalizer function that will release references to
        related compiled functions.
        """
        overloads = self.overloads
        targetctx = self.targetctx

        # Early-bind utils.shutting_down() into the function's local namespace
        # (see issue #689)
        def finalizer(shutting_down=utils.shutting_down):
            # The finalizer may crash at shutdown, skip it (resources
            # will be cleared by the process exiting, anyway).
            if shutting_down():
                return
            # This function must *not* hold any reference to self:
            # we take care to bind the necessary objects in the closure.
            for cres in overloads.values():
                try:
                    targetctx.remove_user_function(cres.entry_point)
                except KeyError:
                    pass

        return finalizer

    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
        return list(self.overloads)

    @property
    def nopython_signatures(self):
        return [
            cres.signature
            for cres in self.overloads.values()
            if not cres.objectmode
        ]

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time."""
        # If disabling compilation then there must be at least one signature
        assert (not val) or len(self.signatures) > 0
        self._can_compile = not val

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        sig = [a._code for a in args]
        self._insert(sig, cres.entry_point, cres.objectmode)
        self.overloads[args] = cres

    def fold_argument_types(self, args, kws):
        return self._compiler.fold_argument_types(args, kws)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows to resolve the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # XXX how about a dispatcher template class automating the
        # following?

        # Fold keyword arguments and resolve default values
        pysig, args = self._compiler.fold_argument_types(args, kws)
        kws = {}
        # Ensure an overload is available
        if self._can_compile:
            self.compile(tuple(args))

        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        # The `key` isn't really used except for diagnosis here,
        # so avoid keeping a reference to `cfunc`.
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures
        )
        return call_template, pysig, args, kws

    def get_overload(self, sig):
        """
        Return the compiled function for the given signature.
        """
        args, return_type = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)].entry_point

    @property
    def is_compiling(self):
        """
        Whether a specialization is currently being compiled.
        """
        return self._compiling_counter

    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
        assert not kws
        # call any initialisation required for the compilation chain (e.g.
        # extension point registration).
        self._compilation_chain_init_hook()

        def error_rewrite(e, issue_type):
            """
            Rewrite and raise Exception `e` with help supplied based on the
            specified issue_type.
            """
            if config.SHOW_HELP:
                help_msg = errors.error_extras[issue_type]
                e.patch_message("\n".join((str(e).rstrip(), help_msg)))
            if config.FULL_TRACEBACKS:
                raise e
            else:
                raise e.with_traceback(None)

        argtypes = []
        for a in args:
            if isinstance(a, OmittedArg):
                argtypes.append(types.Omitted(a.value))
            else:
                argtypes.append(self.typeof_pyval(a))

        return_val = None
        try:
            return_val = self.compile(tuple(argtypes))
        except errors.ForceLiteralArg as e:
            # Received request for compiler re-entry with the list of arguments
            # indicated by e.requested_args.
            # First, check if any of these args are already Literal-ized
            already_lit_pos = [
                i
                for i in e.requested_args
                if isinstance(args[i], types.Literal)
            ]
            if already_lit_pos:
                # Abort compilation if any argument is already a Literal.
                # Letting this continue will cause infinite compilation loop.
                m = (
                    "Repeated literal typing request.\n"
                    "{}.\n"
                    "This is likely caused by an error in typing. "
                    "Please see nested and suppressed exceptions."
                )
                info = ", ".join(
                    "Arg #{} is {}".format(i, args[i])
                    for i in sorted(already_lit_pos)
                )
                raise errors.CompilerError(m.format(info))
            # Convert requested arguments into a Literal.
            args = [
                (types.literal if i in e.requested_args else lambda x: x)(
                    args[i]
                )
                for i, v in enumerate(args)
            ]
            # Re-enter compilation with the Literal-ized arguments
            return_val = self._compile_for_args(*args)

        except errors.TypingError as e:
            # Intercept typing error that may be due to an argument
            # that failed inferencing as a Numba type
            failed_args = []
            for i, arg in enumerate(args):
                val = arg.value if isinstance(arg, OmittedArg) else arg
                try:
                    tp = typeof(val, Purpose.argument)
                except ValueError as typeof_exc:
                    failed_args.append((i, str(typeof_exc)))
                else:
                    if tp is None:
                        failed_args.append(
                            (i, f"cannot determine Numba type of value {val}")
                        )
            if failed_args:
                # Patch error message to ease debugging
                args_str = "\n".join(
                    f"- argument {i}: {err}" for i, err in failed_args
                )
                msg = (
                    f"{str(e).rstrip()} \n\nThis error may have been caused "
                    f"by the following argument(s):\n{args_str}\n"
                )
                e.patch_message(msg)

            error_rewrite(e, "typing")
        except errors.UnsupportedError as e:
            # Something unsupported is present in the user code, add help info
            error_rewrite(e, "unsupported_error")
        except (
            errors.NotDefinedError,
            errors.RedefinedError,
            errors.VerificationError,
        ) as e:
            # These errors are probably from an issue with either the code
            # supplied being syntactically or otherwise invalid
            error_rewrite(e, "interpreter")
        except errors.ConstantInferenceError as e:
            # this is from trying to infer something as constant when it isn't
            # or isn't supported as a constant
            error_rewrite(e, "constant_inference")
        except Exception as e:
            if config.SHOW_HELP:
                if hasattr(e, "patch_message"):
                    help_msg = errors.error_extras["reportable"]
                    e.patch_message("\n".join((str(e).rstrip(), help_msg)))
            # ignore the FULL_TRACEBACKS config, this needs reporting!
            raise e
        finally:
            self._types_active_call.clear()
        return return_val

    def inspect_llvm(self, signature=None):
        """Get the LLVM intermediate representation generated by compilation.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the LLVM IR. If None, the
            IR is returned for all available signatures.

        Returns
        -------
        llvm : dict[signature, str] or str
            Either the LLVM IR string for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to LLVM IR
            strings.
        """
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_llvm_str()

        return dict((sig, self.inspect_llvm(sig)) for sig in self.signatures)

    def inspect_asm(self, signature=None):
        """Get the generated assembly code.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the assembly code. If
            None, the assembly code is returned for all available signatures.

        Returns
        -------
        asm : dict[signature, str] or str
            Either the assembly code for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to assembly
            code.
        """
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_asm_str()

        return dict((sig, self.inspect_asm(sig)) for sig in self.signatures)

    def inspect_types(
        self, file=None, signature=None, pretty=False, style="default", **kwargs
    ):
        """Print/return Numba intermediate representation (IR)-annotated code.

        Parameters
        ----------
        file : file-like object, optional
            File to which to print. Defaults to sys.stdout if None. Must be
            None if ``pretty=True``.
        signature : tuple of numba types, optional
            Print/return the intermediate representation for only the given
            signature. If None, the IR is printed for all available signatures.
        pretty : bool, optional
            If True, an Annotate object will be returned that can render the
            IR with color highlighting in Jupyter and IPython. ``file`` must
            be None if ``pretty`` is True. Additionally, the ``pygments``
            library must be installed for ``pretty=True``.
        style : str, optional
            Choose a style for rendering. Ignored if ``pretty`` is ``False``.
            This is directly consumed by ``pygments`` formatters. To see a
            list of available styles, import ``pygments`` and run
            ``list(pygments.styles.get_all_styles())``.

        Returns
        -------
        annotated : Annotate object, optional
            Only returned if ``pretty=True``, otherwise this function is only
            used for its printing side effect. If ``pretty=True``, an Annotate
            object is returned that can render itself in Jupyter and IPython.
        """
        overloads = self.overloads
        if signature is not None:
            overloads = {signature: self.overloads[signature]}

        if not pretty:
            if file is None:
                file = sys.stdout

            for ver, res in overloads.items():
                print("%s %s" % (self.py_func.__name__, ver), file=file)
                print("-" * 80, file=file)
                print(res.type_annotation, file=file)
                print("=" * 80, file=file)
        else:
            if file is not None:
                raise ValueError("`file` must be None if `pretty=True`")
            from numba.core.annotations.pretty_annotate import Annotate

            return Annotate(self, signature=signature, style=style)

    def inspect_cfg(self, signature=None, show_wrapper=None, **kwargs):
        """
        For inspecting the CFG of the function.

        By default the CFG of the user function is shown.  The *show_wrapper*
        option can be set to "python" or "cfunc" to show the python wrapper
        function or the *cfunc* wrapper function, respectively.

        Parameters accepted in kwargs
        -----------------------------
        filename : string, optional
            the name of the output file, if given this will write the output to
            filename
        view : bool, optional
            whether to immediately view the optional output file
        highlight : bool, set, dict, optional
            what, if anything, to highlight, options are:
            { incref : bool, # highlight NRT_incref calls
              decref : bool, # highlight NRT_decref calls
              returns : bool, # highlight exits which are normal returns
              raises : bool, # highlight exits which are from raise
              meminfo : bool, # highlight calls to NRT*meminfo
              branches : bool, # highlight true/false branches
             }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'incref', 'decref'} would
            switch on highlighting on increfs and decrefs.
        interleave: bool, set, dict, optional
            what, if anything, to interleave in the LLVM IR, options are:
            { python: bool # interleave python source code with the LLVM IR
              lineinfo: bool # interleave line information markers with the LLVM
                             # IR
            }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {'python',} would
            switch on interleaving of python source code in the LLVM IR.
        strip_ir : bool, optional
            Default is False. If set to True all LLVM IR that is superfluous to
            that requested in kwarg `highlight` will be removed.
        show_key : bool, optional
            Default is True. Create a "key" for the highlighting in the rendered
            CFG.
        fontsize : int, optional
            Default is 8. Set the fontsize in the output to this value.
        """
        if signature is not None:
            cres = self.overloads[signature]
            lib = cres.library
            if show_wrapper == "python":
                fname = cres.fndesc.llvm_cpython_wrapper_name
            elif show_wrapper == "cfunc":
                fname = cres.fndesc.llvm_cfunc_wrapper_name
            else:
                fname = cres.fndesc.mangled_name
            return lib.get_function_cfg(fname, py_func=self.py_func, **kwargs)

        return dict(
            (sig, self.inspect_cfg(sig, show_wrapper=show_wrapper))
            for sig in self.signatures
        )

    def inspect_disasm_cfg(self, signature=None):
        """
        For inspecting the CFG of the disassembly of the function.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz

        signature : tuple of Numba types, optional
            Print/return the disassembly CFG for only the given signatures.
            If None, the IR is printed for all available signatures.
        """
        if signature is not None:
            cres = self.overloads[signature]
            lib = cres.library
            return lib.get_disasm_cfg(cres.fndesc.mangled_name)

        return dict(
            (sig, self.inspect_disasm_cfg(sig)) for sig in self.signatures
        )

    def get_annotation_info(self, signature=None):
        """
        Gets the annotation information for the function specified by
        signature. If no signature is supplied a dictionary of signature to
        annotation information is returned.
        """
        signatures = self.signatures if signature is None else [signature]
        out = collections.OrderedDict()
        for sig in signatures:
            cres = self.overloads[sig]
            ta = cres.type_annotation
            key = (
                ta.func_id.filename + ":" + str(ta.func_id.firstlineno + 1),
                ta.signature,
            )
            out[key] = ta.annotate_raw()[key]
        return out

    def _explain_ambiguous(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, "kwargs not handled"
        args = tuple([self.typeof_pyval(a) for a in args])
        # The order here must be deterministic for testing purposes, which
        # is ensured by the OrderedDict.
        sigs = self.nopython_signatures
        # This will raise
        self.typingctx.resolve_overload(
            self.py_func, sigs, args, kws, allow_ambiguous=False
        )

    def _explain_matching_error(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, "kwargs not handled"
        args = [self.typeof_pyval(a) for a in args]
        msg = "No matching definition for argument type(s) %s" % ", ".join(
            map(str, args)
        )
        raise TypeError(msg)

    def _search_new_conversions(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        Search for approximately matching signatures for the given arguments,
        and ensure the corresponding conversions are registered in the C++
        type manager.
        """
        assert not kws, "kwargs not handled"
        args = [self.typeof_pyval(a) for a in args]
        found = False
        for sig in self.nopython_signatures:
            conv = self.typingctx.install_possible_conversions(args, sig.args)
            if conv:
                found = True
        return found

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.py_func)

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        try:
            tp = typeof(val, Purpose.argument)
        except ValueError:
            tp = types.pyobject
        else:
            if tp is None:
                tp = types.pyobject
        self._types_active_call.add(tp)
        return tp

    def _callback_add_timer(self, duration, cres, lock_name):
        md = cres.metadata
        # md can be None when code is loaded from cache
        if md is not None:
            timers = md.setdefault("timers", {})
            if lock_name not in timers:
                # Only write if the metadata does not exist
                timers[lock_name] = duration
            else:
                msg = f"'{lock_name} metadata is already defined."
                raise AssertionError(msg)

    def _callback_add_compiler_timer(self, duration, cres):
        return self._callback_add_timer(
            duration, cres, lock_name="compiler_lock"
        )

    def _callback_add_llvm_timer(self, duration, cres):
        return self._callback_add_timer(duration, cres, lock_name="llvm_lock")


class _MemoMixin:
    __uuid = None
    # A {uuid -> instance} mapping, for deserialization
    _memo = weakref.WeakValueDictionary()
    # hold refs to last N functions deserialized, retaining them in _memo
    # regardless of whether there is another reference
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)

    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note: this is lazily-generated, for performance reasons.
        """
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid4())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self
        self._recent.append(self)


_CompileStats = collections.namedtuple(
    "_CompileStats", ("cache_path", "cache_hits", "cache_misses")
)


class _FunctionCompiler(object):
    def __init__(self, py_func, targetdescr, targetoptions, pipeline_class):
        self.py_func = py_func
        self.targetdescr = targetdescr
        self.targetoptions = targetoptions
        self.locals = {}
        self.pysig = utils.pysignature(self.py_func)
        self.pipeline_class = pipeline_class
        # Remember key=(args, return_type) combinations that will fail
        # compilation to avoid compilation attempt on them.  The values are
        # the exceptions.
        self._failed_cache = {}

    def fold_argument_types(self, args, kws):
        """
        Given positional and named argument types, fold keyword arguments
        and resolve defaults by inserting types.Omitted() instances.

        A (pysig, argument types) tuple is returned.
        """

        def normal_handler(index, param, value):
            return value

        def default_handler(index, param, default):
            return types.Omitted(default)

        def stararg_handler(index, param, values):
            return types.StarArgTuple(values)

        # For now, we take argument values from the @jit function
        args = fold_arguments(
            self.pysig,
            args,
            kws,
            normal_handler,
            default_handler,
            stararg_handler,
        )
        return self.pysig, args

    def compile(self, args, return_type):
        status, retval = self._compile_cached(args, return_type)
        if status:
            return retval
        else:
            raise retval

    def _compile_cached(self, args, return_type):
        key = tuple(args), return_type
        try:
            return False, self._failed_cache[key]
        except KeyError:
            pass

        try:
            retval = self._compile_core(args, return_type)
        except errors.TypingError as e:
            self._failed_cache[key] = e
            return False, e
        else:
            return True, retval

    def _compile_core(self, args, return_type):
        flags = Flags()
        self.targetdescr.options.parse_as_flags(flags, self.targetoptions)
        flags = self._customize_flags(flags)

        impl = self._get_implementation(args, {})
        cres = compile_extra(
            self.targetdescr.typing_context,
            self.targetdescr.target_context,
            impl,
            args=args,
            return_type=return_type,
            flags=flags,
            locals=self.locals,
            pipeline_class=self.pipeline_class,
        )
        # Check typing error if object mode is used
        if cres.typing_error is not None and not flags.enable_pyobject:
            raise cres.typing_error
        return cres

    def get_globals_for_reduction(self):
        return serialize._get_function_globals_for_reduction(self.py_func)

    def _get_implementation(self, args, kws):
        return self.py_func

    def _customize_flags(self, flags):
        return flags


class CUDADispatcher(serialize.ReduceMixin, _MemoMixin, _DispatcherBase):
    """
    CUDA Dispatcher object. When configured and called, the dispatcher will
    specialize itself for the given arguments (if no suitable specialized
    version already exists) & compute capability, and launch on the device
    associated with the current context.

    Dispatcher objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    """

    # Whether to fold named arguments and default values. Default values are
    # presently unsupported on CUDA, so we can leave this as False in all
    # cases.
    _fold_args = False

    targetdescr = cuda_target

    def __init__(self, py_func, targetoptions, pipeline_class=CUDACompiler):
        """
        Parameters
        ----------
        py_func: function object to be compiled
        targetoptions: dict, optional
            Target-specific config options.
        pipeline_class: type numba.compiler.CompilerBase
            The compiler pipeline type.
        """
        self.typingctx = self.targetdescr.typing_context
        self.targetctx = self.targetdescr.target_context

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)
        can_fallback = not targetoptions.get("nopython", False)

        _DispatcherBase.__init__(
            self,
            arg_count,
            py_func,
            pysig,
            can_fallback,
            exact_match_required=False,
        )

        functools.update_wrapper(self, py_func)

        self.targetoptions = targetoptions
        self._cache = NullCache()
        compiler_class = _FunctionCompiler
        self._compiler = compiler_class(
            py_func, self.targetdescr, targetoptions, pipeline_class
        )
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()

        # The following properties are for specialization of CUDADispatchers. A
        # specialized CUDADispatcher is one that is compiled for exactly one
        # set of argument types, and bypasses some argument type checking for
        # faster kernel launches.

        # Is this a specialized dispatcher?
        self._specialized = False

        # If we produced specialized dispatchers, we cache them for each set of
        # argument types
        self.specializations = {}

    def dump(self, tab=""):
        print(
            f"{tab}DUMP {type(self).__name__}[{self.py_func.__name__}"
            f", type code={self._type._code}]"
        )
        for cres in self.overloads.values():
            cres.dump(tab=tab + "  ")
        print(f"{tab}END DUMP {type(self).__name__}[{self.py_func.__name__}]")

    @property
    def _numba_type_(self):
        return cuda_types.CUDADispatcher(self)

    def enable_caching(self):
        self._cache = CUDACache(self.py_func)

    def __get__(self, obj, objtype=None):
        """Allow a JIT function to be bound as a method to an object"""
        if obj is None:  # Unbound method
            return self
        else:  # Bound method
            return pytypes.MethodType(self, obj)

    @functools.lru_cache(maxsize=128)
    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        return _LaunchConfiguration(self, griddim, blockdim, stream, sharedmem)

    def __getitem__(self, args):
        if len(args) not in [2, 3, 4]:
            raise ValueError("must specify at least the griddim and blockdim")
        return self.configure(*args)

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        """Returns a 1D-configured dispatcher for a given number of tasks.

        This assumes that:

        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
          1-1 basis.
        - the kernel checks that the Global Thread ID is upper-bounded by
          ``ntasks``, and does nothing if it is not.

        :param ntasks: The number of tasks.
        :param tpb: The size of a block. An appropriate value is chosen if this
                    parameter is not supplied.
        :param stream: The stream on which the configured dispatcher will be
                       launched.
        :param sharedmem: The number of bytes of dynamic shared memory required
                          by the kernel.
        :return: A configured dispatcher, ready to launch on a set of
                 arguments."""

        return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)

    @property
    def extensions(self):
        """
        A list of objects that must have a `prepare_args` function. When a
        specialized kernel is called, each argument will be passed through
        to the `prepare_args` (from the last object in this list to the
        first). The arguments to `prepare_args` are:

        - `ty` the numba type of the argument
        - `val` the argument value itself
        - `stream` the CUDA stream used for the current call to the kernel
        - `retr` a list of zero-arg functions that you may want to append
          post-call cleanup work to.

        The `prepare_args` function must return a tuple `(ty, val)`, which
        will be passed in turn to the next right-most `extension`. After all
        the extensions have been called, the resulting `(ty, val)` will be
        passed into Numba's default argument marshalling logic.
        """
        return self.targetoptions.get("extensions")

    def __call__(self, *args, **kwargs):
        # An attempt to launch an unconfigured kernel
        raise ValueError(missing_launch_config_msg)

    def call(self, args, griddim, blockdim, stream, sharedmem):
        """
        Compile if necessary and invoke this kernel with *args*.
        """
        if self.specialized:
            kernel = next(iter(self.overloads.values()))
        else:
            kernel = _dispatcher.Dispatcher._cuda_call(self, *args)

        kernel.launch(args, griddim, blockdim, stream, sharedmem)

    def _compile_for_args(self, *args, **kws):
        # Based on _DispatcherBase._compile_for_args.
        assert not kws
        argtypes = [self.typeof_pyval(a) for a in args]
        return self.compile(tuple(argtypes))

    def typeof_pyval(self, val):
        # Based on _DispatcherBase.typeof_pyval, but differs from it to support
        # the CUDA Array Interface.
        try:
            return typeof(val, Purpose.argument)
        except ValueError:
            if cuda.is_cuda_array(val):
                # When typing, we don't need to synchronize on the array's
                # stream - this is done when the kernel is launched.
                return typeof(
                    cuda.as_cuda_array(val, sync=False), Purpose.argument
                )
            else:
                raise

    def specialize(self, *args):
        """
        Create a new instance of this dispatcher specialized for the given
        *args*.
        """
        cc = get_current_device().compute_capability
        argtypes = tuple(self.typeof_pyval(a) for a in args)
        if self.specialized:
            raise RuntimeError("Dispatcher already specialized")

        specialization = self.specializations.get((cc, argtypes))
        if specialization:
            return specialization

        targetoptions = self.targetoptions
        specialization = CUDADispatcher(
            self.py_func, targetoptions=targetoptions
        )
        specialization.compile(argtypes)
        specialization.disable_compile()
        specialization._specialized = True
        self.specializations[cc, argtypes] = specialization
        return specialization

    @property
    def specialized(self):
        """
        True if the Dispatcher has been specialized.
        """
        return self._specialized

    def get_regs_per_thread(self, signature=None):
        """
        Returns the number of registers used by each thread in this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get register
                          usage for. This may be omitted for a specialized
                          kernel.
        :return: The number of registers used by the compiled variant of the
                 kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].regs_per_thread
        if self.specialized:
            return next(iter(self.overloads.values())).regs_per_thread
        else:
            return {
                sig: overload.regs_per_thread
                for sig, overload in self.overloads.items()
            }

    def get_const_mem_size(self, signature=None):
        """
        Returns the size in bytes of constant memory used by this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get constant
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The size in bytes of constant memory allocated by the
                 compiled variant of the kernel for the given signature and
                 current device.
        """
        if signature is not None:
            return self.overloads[signature.args].const_mem_size
        if self.specialized:
            return next(iter(self.overloads.values())).const_mem_size
        else:
            return {
                sig: overload.const_mem_size
                for sig, overload in self.overloads.items()
            }

    def get_shared_mem_per_block(self, signature=None):
        """
        Returns the size in bytes of statically allocated shared memory
        for this kernel.

        :param signature: The signature of the compiled kernel to get shared
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of shared memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].shared_mem_per_block
        if self.specialized:
            return next(iter(self.overloads.values())).shared_mem_per_block
        else:
            return {
                sig: overload.shared_mem_per_block
                for sig, overload in self.overloads.items()
            }

    def get_max_threads_per_block(self, signature=None):
        """
        Returns the maximum allowable number of threads per block
        for this kernel. Exceeding this threshold will result in
        the kernel failing to launch.

        :param signature: The signature of the compiled kernel to get the max
                          threads per block for. This may be omitted for a
                          specialized kernel.
        :return: The maximum allowable threads per block for the compiled
                 variant of the kernel for the given signature and current
                 device.
        """
        if signature is not None:
            return self.overloads[signature.args].max_threads_per_block
        if self.specialized:
            return next(iter(self.overloads.values())).max_threads_per_block
        else:
            return {
                sig: overload.max_threads_per_block
                for sig, overload in self.overloads.items()
            }

    def get_local_mem_per_thread(self, signature=None):
        """
        Returns the size in bytes of local memory per thread
        for this kernel.

        :param signature: The signature of the compiled kernel to get local
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of local memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].local_mem_per_thread
        if self.specialized:
            return next(iter(self.overloads.values())).local_mem_per_thread
        else:
            return {
                sig: overload.local_mem_per_thread
                for sig, overload in self.overloads.items()
            }

    def get_call_template(self, args, kws):
        # Originally copied from _DispatcherBase.get_call_template. This
        # version deviates slightly from the _DispatcherBase version in order
        # to force casts when calling device functions. See e.g.
        # TestDeviceFunc.test_device_casting, added in PR #7496.
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows resolution of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # Fold keyword arguments and resolve default values
        pysig, args = self.fold_argument_types(args, kws)
        kws = {}

        # Ensure an exactly-matching overload is available if we can
        # compile. We proceed with the typing even if we can't compile
        # because we may be able to force a cast on the caller side.
        if self._can_compile:
            self.compile_device(tuple(args))

        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)

        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures
        )
        pysig = utils.pysignature(self.py_func)

        return call_template, pysig, args, kws

    def compile_device(self, args, return_type=None):
        """Compile the device function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
        if args not in self.overloads:
            with self._compiling_counter:
                debug = self.targetoptions.get("debug")
                lineinfo = self.targetoptions.get("lineinfo")
                forceinline = self.targetoptions.get("forceinline")
                fastmath = self.targetoptions.get("fastmath")

                nvvm_options = {
                    "opt": 3 if self.targetoptions.get("opt") else 0,
                    "fastmath": fastmath,
                }

                if debug:
                    nvvm_options["g"] = None

                cc = get_current_device().compute_capability
                cres = compile_cuda(
                    self.py_func,
                    return_type,
                    args,
                    debug=debug,
                    lineinfo=lineinfo,
                    forceinline=forceinline,
                    fastmath=fastmath,
                    nvvm_options=nvvm_options,
                    cc=cc,
                )
                self.overloads[args] = cres

                cres.target_context.insert_user_function(
                    cres.entry_point, cres.fndesc, [cres.library]
                )
        else:
            cres = self.overloads[args]

        return cres

    def add_overload(self, kernel, argtypes):
        c_sig = [a._code for a in argtypes]
        self._insert(c_sig, kernel, cuda=True)
        self.overloads[argtypes] = kernel

    @global_compiler_lock
    def compile(self, sig):
        """
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        """
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None or return_type == types.none

        # Do we already have an in-memory compiled kernel?
        if self.specialized:
            return next(iter(self.overloads.values()))
        else:
            kernel = self.overloads.get(argtypes)
            if kernel is not None:
                return kernel

        # Can we load from the disk cache?
        kernel = self._cache.load_overload(sig, self.targetctx)

        if kernel is not None:
            self._cache_hits[sig] += 1
        else:
            # We need to compile a new kernel
            self._cache_misses[sig] += 1
            if not self._can_compile:
                raise RuntimeError("Compilation disabled")

            kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
            # We call bind to force codegen, so that there is a cubin to cache
            kernel.bind()
            self._cache.save_overload(sig, kernel)

        self.add_overload(kernel, argtypes)

        return kernel

    def get_compile_result(self, sig):
        """Compile (if needed) and return the compilation result with the
        given signature.

        Returns ``CompileResult``.
        Raises ``NumbaError`` if the signature is incompatible.
        """
        atypes = tuple(sig.args)
        if atypes not in self.overloads:
            if self._can_compile:
                # Compiling may raise any NumbaError
                self.compile(atypes)
            else:
                msg = f"{sig} not available and compilation disabled"
                raise errors.TypingError(msg)
        return self.overloads[atypes]

    def recompile(self):
        """
        Recompile all signatures afresh.
        """
        sigs = list(self.overloads)
        old_can_compile = self._can_compile
        # Ensure the old overloads are disposed of,
        # including compiled functions.
        self._make_finalizer()()
        self._reset_overloads()
        self._cache.flush()
        self._can_compile = True
        try:
            for sig in sigs:
                self.compile(sig)
        finally:
            self._can_compile = old_can_compile

    @property
    def stats(self):
        return _CompileStats(
            cache_path=self._cache.cache_path,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

    def get_metadata(self, signature=None):
        """
        Obtain the compilation metadata for a given signature.
        """
        if signature is not None:
            return self.overloads[signature].metadata
        else:
            return dict(
                (sig, self.overloads[sig].metadata) for sig in self.signatures
            )

    def get_function_type(self):
        """Return unique function type of dispatcher when possible, otherwise
        return None.

        A Dispatcher instance has unique function type when it
        contains exactly one compilation result and its compilation
        has been disabled (via its disable_compile method).
        """
        if not self._can_compile and len(self.overloads) == 1:
            cres = tuple(self.overloads.values())[0]
            return types.FunctionType(cres.signature)

    def inspect_llvm(self, signature=None):
        """
        Return the LLVM IR for this kernel.

        :param signature: A tuple of argument types.
        :return: The LLVM IR for the given signature, or a dict of LLVM IR
                 for all previously-encountered signatures.

        """
        device = self.targetoptions.get("device")
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_llvm_str()
            else:
                return self.overloads[signature].inspect_llvm()
        else:
            if device:
                return {
                    sig: overload.library.get_llvm_str()
                    for sig, overload in self.overloads.items()
                }
            else:
                return {
                    sig: overload.inspect_llvm()
                    for sig, overload in self.overloads.items()
                }

    def inspect_asm(self, signature=None):
        """
        Return this kernel's PTX assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The PTX code for the given signature, or a dict of PTX codes
                 for all previously-encountered signatures.
        """
        cc = get_current_device().compute_capability
        device = self.targetoptions.get("device")
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_asm_str(cc)
            else:
                return self.overloads[signature].inspect_asm(cc)
        else:
            if device:
                return {
                    sig: overload.library.get_asm_str(cc)
                    for sig, overload in self.overloads.items()
                }
            else:
                return {
                    sig: overload.inspect_asm(cc)
                    for sig, overload in self.overloads.items()
                }

    def inspect_lto_ptx(self, signature=None):
        """
        Return link-time optimized PTX code for the given signature.

        :param signature: A tuple of argument types.
        :return: The PTX code for the given signature, or a dict of PTX codes
                 for all previously-encountered signatures.
        """
        cc = get_current_device().compute_capability
        device = self.targetoptions.get("device")

        if signature is not None:
            if device:
                return self.overloads[signature].library.get_lto_ptx(cc)
            else:
                return self.overloads[signature].inspect_lto_ptx(cc)
        else:
            if device:
                return {
                    sig: overload.library.get_lto_ptx(cc)
                    for sig, overload in self.overloads.items()
                }
            else:
                return {
                    sig: overload.inspect_lto_ptx(cc)
                    for sig, overload in self.overloads.items()
                }

    def inspect_sass_cfg(self, signature=None):
        """
        Return this kernel's CFG for the device in the current context.

        :param signature: A tuple of argument types.
        :return: The CFG for the given signature, or a dict of CFGs
                 for all previously-encountered signatures.

        The CFG for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
        if self.targetoptions.get("device"):
            raise RuntimeError("Cannot get the CFG of a device function")

        if signature is not None:
            return self.overloads[signature].inspect_sass_cfg()
        else:
            return {
                sig: defn.inspect_sass_cfg()
                for sig, defn in self.overloads.items()
            }

    def inspect_sass(self, signature=None):
        """
        Return this kernel's SASS assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The SASS code for the given signature, or a dict of SASS codes
                 for all previously-encountered signatures.

        SASS for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
        if self.targetoptions.get("device"):
            raise RuntimeError("Cannot inspect SASS of a device function")

        if signature is not None:
            return self.overloads[signature].inspect_sass()
        else:
            return {
                sig: defn.inspect_sass() for sig, defn in self.overloads.items()
            }

    def inspect_types(self, file=None):
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
        if file is None:
            file = sys.stdout

        for _, defn in self.overloads.items():
            defn.inspect_types(file=file)

    @classmethod
    def _rebuild(cls, py_func, targetoptions):
        """
        Rebuild an instance.
        """
        instance = cls(py_func, targetoptions)
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
        return dict(py_func=self.py_func, targetoptions=self.targetoptions)


# Initialize typeof machinery
_dispatcher.typeof_init(
    OmittedArg, dict((str(t), t._code) for t in types.number_domain)
)
