# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from llvmlite import ir

from numba.cuda.core import config
from numba.cuda import serialize
from .cudadrv import devices, driver, nvvm, runtime, nvrtc
from numba.cuda.core.codegen import Codegen, CodeLibrary
from numba.cuda.cudadrv.libs import get_cudalib
from numba.cuda.cudadrv.linkable_code import LinkableCode
from numba.cuda.memory_management.nrt import NRT_LIBRARY

import os
import subprocess
import tempfile

CUDA_TRIPLE = "nvptx64-nvidia-cuda"


def run_nvdisasm(cubin, flags):
    # nvdisasm only accepts input from a file, so we need to write out to a
    # temp file and clean up afterwards.
    fd = None
    fname = None
    try:
        fd, fname = tempfile.mkstemp()
        with open(fname, "wb") as f:
            f.write(cubin.code)

        try:
            cp = subprocess.run(
                ["nvdisasm", *flags, fname],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            msg = (
                "nvdisasm has not been found. You may need "
                "to install the CUDA toolkit and ensure that "
                "it is available on your PATH.\n"
            )
            raise RuntimeError(msg) from e
        return cp.stdout.decode("utf-8")
    finally:
        if fd is not None:
            os.close(fd)
        if fname is not None:
            os.unlink(fname)


def disassemble_cubin(cubin):
    # Request lineinfo in disassembly
    flags = ["-gi"]
    return run_nvdisasm(cubin, flags)


def disassemble_cubin_for_cfg(cubin):
    # Request control flow graph in disassembly
    flags = ["-cfg"]
    return run_nvdisasm(cubin, flags)


class ExternalCodeLibrary(CodeLibrary):
    """Holds code produced externally, for linking with generated code."""

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        # Files to link
        self._linking_files = set()
        # Setup and teardown functions for the module.
        # The order is determined by the order they are added to the codelib.
        self._setup_functions = []
        self._teardown_functions = []

        self.use_cooperative = False

    @property
    def modules(self):
        # There are no LLVM IR modules in an ExternalCodeLibrary
        return set()

    def add_linking_file(self, path_or_obj):
        # Adding new files after finalization is prohibited, in case the list
        # of libraries has already been added to another code library; the
        # newly-added files would be omitted from their linking process.
        self._raise_if_finalized()

        if isinstance(path_or_obj, LinkableCode):
            if path_or_obj.setup_callback:
                self._setup_functions.append(path_or_obj.setup_callback)
            if path_or_obj.teardown_callback:
                self._teardown_functions.append(path_or_obj.teardown_callback)

        self._linking_files.add(path_or_obj)

    def add_ir_module(self, module):
        raise NotImplementedError("Cannot add LLVM IR to external code")

    def add_linking_library(self, library):
        raise NotImplementedError("Cannot add libraries to external code")

    def finalize(self):
        self._raise_if_finalized()
        self._finalized = True

    def get_asm_str(self):
        raise NotImplementedError("No assembly for external code")

    def get_llvm_str(self):
        raise NotImplementedError("No LLVM IR for external code")

    def get_function(self, name):
        raise NotImplementedError("Cannot get function from external code")


class CUDACodeLibrary(serialize.ReduceMixin, CodeLibrary):
    """
    The CUDACodeLibrary generates PTX, SASS, cubins for multiple different
    compute capabilities. It also loads cubins to multiple devices (via
    get_cufunc), which may be of different compute capabilities.
    """

    def __init__(
        self,
        codegen,
        name,
        entry_name=None,
        max_registers=None,
        lto=False,
        nvvm_options=None,
    ):
        """
        codegen:
            Codegen object.
        name:
            Name of the function in the source.
        entry_name:
            Name of the kernel function in the binary, if this is a global
            kernel and not a device function.
        max_registers:
            The maximum register usage to aim for when linking.
        lto:
            Whether to enable link-time optimization.
        nvvm_options:
                Dict of options to pass to NVVM.
        """
        super().__init__(codegen, name)

        # The llvmlite module for this library.
        self._module = None
        # CodeLibrary objects that will be "linked" into this library. The
        # modules within them are compiled from NVVM IR to PTX along with the
        # IR from this module - in that sense they are "linked" by NVVM at PTX
        # generation time, rather than at link time.
        self._linking_libraries = set()
        # Files to link with the generated PTX. These are linked using the
        # Driver API at link time.
        self._linking_files = set()
        # List of setup functions to the loaded module
        # the order is determined by the order they are added to the codelib.
        self._setup_functions = []
        # List of teardown functions to the loaded module
        # the order is determined by the order they are added to the codelib.
        self._teardown_functions = []
        # Should we link libcudadevrt?
        self.needs_cudadevrt = False

        # Cache the LLVM IR string
        self._llvm_strs = None
        # Maps CC -> PTX string
        self._ptx_cache = {}
        # Maps CC -> LTO-IR
        self._ltoir_cache = {}
        # Maps CC -> cubin
        self._cubin_cache = {}
        # Maps CC -> linker info output for cubin
        self._linkerinfo_cache = {}
        # Maps Device numeric ID -> cufunc
        self._cufunc_cache = {}

        self._max_registers = max_registers
        self._lto = lto
        if nvvm_options is None:
            nvvm_options = {}
        self._nvvm_options = nvvm_options
        self._entry_name = entry_name

        self.use_cooperative = False

        # Objects that need to be kept alive for the lifetime of the
        # kernels or device functions generated by this code library,
        # e.g., device arrays captured from global scope.
        self.referenced_objects = {}

    @property
    def llvm_strs(self):
        if self._llvm_strs is None:
            self._llvm_strs = [str(mod) for mod in self.modules]
        return self._llvm_strs

    def get_llvm_str(self):
        return "\n\n".join(self.llvm_strs)

    def _ensure_cc(self, cc):
        if cc is not None:
            return cc

        device = devices.get_context().device
        return device.compute_capability

    def get_asm_str(self, cc=None):
        cc = self._ensure_cc(cc)

        ptxes = self._ptx_cache.get(cc, None)
        if ptxes:
            return ptxes

        arch = nvrtc.get_arch_option(*cc)
        options = self._nvvm_options.copy()
        options["arch"] = arch

        irs = self.llvm_strs

        ptx = nvvm.compile_ir(irs, **options)

        # Sometimes the result from NVVM contains trailing whitespace and
        # nulls, which we strip so that the assembly dump looks a little
        # tidier.
        ptx = ptx.decode().strip("\x00").strip()

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % self._name).center(80, "-"))
            print(ptx)
            print("=" * 80)

        self._ptx_cache[cc] = ptx

        return ptx

    def get_lto_ptx(self, cc=None):
        """
        Get the PTX code after LTO.
        """

        if not self._lto:
            raise RuntimeError("LTO is not enabled")

        if not driver._have_nvjitlink():
            raise RuntimeError("Link time optimization requires nvJitLink.")

        cc = self._ensure_cc(cc)

        linker = driver._Linker(
            max_registers=self._max_registers,
            cc=cc,
            additional_flags=["-ptx"],
            lto=self._lto,
        )

        self._link_all(linker, cc, ignore_nonlto=True)

        ptx = linker.get_linked_ptx()
        ptx = ptx.decode("utf-8")

        return ptx

    def get_ltoir(self, cc=None):
        cc = self._ensure_cc(cc)

        ltoir = self._ltoir_cache.get(cc, None)
        if ltoir is not None:
            return ltoir

        arch = nvrtc.get_arch_option(*cc)
        options = self._nvvm_options.copy()
        options["arch"] = arch
        options["gen-lto"] = None

        irs = self.llvm_strs
        ltoir = nvvm.compile_ir(irs, **options)
        self._ltoir_cache[cc] = ltoir

        return ltoir

    def _link_all(self, linker, cc, ignore_nonlto=False):
        if linker.lto:
            ltoir = self.get_ltoir(cc=cc)
            linker.add_ltoir(ltoir)
        else:
            ptx = self.get_asm_str(cc=cc)
            linker.add_ptx(ptx.encode())

        for path in self._linking_files:
            linker.add_file_guess_ext(path, ignore_nonlto)
        if self.needs_cudadevrt:
            linker.add_file_guess_ext(
                get_cudalib("cudadevrt", static=True), ignore_nonlto
            )

    def get_cubin(self, cc=None):
        cc = self._ensure_cc(cc)

        cubin = self._cubin_cache.get(cc, None)
        if cubin:
            return cubin

        if self._lto and config.DUMP_ASSEMBLY:
            ptx = self.get_lto_ptx(cc=cc)

            print(("ASSEMBLY (AFTER LTO) %s" % self._name).center(80, "-"))
            print(ptx)
            print("=" * 80)

        linker = driver._Linker(
            max_registers=self._max_registers, cc=cc, lto=self._lto
        )
        self._link_all(linker, cc, ignore_nonlto=False)
        cubin = linker.complete()

        self._cubin_cache[cc] = cubin
        self._linkerinfo_cache[cc] = linker.info_log

        return cubin

    def get_cufunc(self):
        if self._entry_name is None:
            msg = (
                "Missing entry_name - are you trying to get the cufunc "
                "for a device function?"
            )
            raise RuntimeError(msg)

        ctx = devices.get_context()
        device = ctx.device

        cufunc = self._cufunc_cache.get(device.id, None)
        if cufunc:
            return cufunc
        cubin = self.get_cubin(cc=device.compute_capability)
        module = ctx.create_module_image(
            cubin, self._setup_functions, self._teardown_functions
        )

        # Load
        cufunc = module.get_function(self._entry_name)

        # Populate caches
        self._cufunc_cache[device.id] = cufunc

        return cufunc

    def get_linkerinfo(self, cc):
        try:
            return self._linkerinfo_cache[cc]
        except KeyError:
            raise KeyError(f"No linkerinfo for CC {cc}")

    def get_sass(self, cc=None):
        return disassemble_cubin(self.get_cubin(cc=cc))

    def get_sass_cfg(self, cc=None):
        return disassemble_cubin_for_cfg(self.get_cubin(cc=cc))

    def add_ir_module(self, mod):
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError("CUDACodeLibrary only supports one module")
        self._module = mod

    def add_linking_library(self, library):
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        self._linking_libraries.add(library)
        self._linking_files.update(library._linking_files)
        self._setup_functions.extend(library._setup_functions)
        self._teardown_functions.extend(library._teardown_functions)
        self.use_cooperative |= library.use_cooperative
        self.referenced_objects.update(getattr(library, "referenced_objects", {}))

    def add_linking_file(self, path_or_obj):
        if isinstance(path_or_obj, LinkableCode):
            if path_or_obj.setup_callback:
                self._setup_functions.append(path_or_obj.setup_callback)
            if path_or_obj.teardown_callback:
                self._teardown_functions.append(path_or_obj.teardown_callback)

        self._linking_files.add(path_or_obj)

    def get_function(self, name):
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f"Function {name} not found")

    @property
    def modules(self):
        return [self._module] + [
            mod for lib in self._linking_libraries for mod in lib.modules
        ]

    @property
    def linking_libraries(self):
        # Libraries we link to may link to other libraries, so we recursively
        # traverse the linking libraries property to build up a list of all
        # linked libraries.
        libs = []
        for lib in self._linking_libraries:
            libs.extend(lib.linking_libraries)
            libs.append(lib)
        return libs

    def finalize(self):
        # Unlike the CPUCodeLibrary, we don't invoke the binding layer here -
        # we only adjust the linkage of functions. Global kernels (with
        # external linkage) have their linkage untouched. Device functions are
        # set linkonce_odr to prevent them appearing in the PTX.

        self._raise_if_finalized()

        # Note in-place modification of the linkage of functions in linked
        # libraries. This presently causes no issues as only device functions
        # are shared across code libraries, so they would always need their
        # linkage set to linkonce_odr. If in a future scenario some code
        # libraries require linkonce_odr linkage of functions in linked
        # modules, and another code library requires another linkage, each code
        # library will need to take its own private copy of its linked modules.
        #
        # See also discussion on PR #890:
        # https://github.com/numba/numba/pull/890
        for library in self._linking_libraries:
            for mod in library.modules:
                for fn in mod.functions:
                    if not fn.is_declaration:
                        fn.linkage = "linkonce_odr"

        self._finalized = True

    def _reduce_states(self):
        """
        Reduce the instance for serialization. We retain the PTX and cubins,
        but loaded functions are discarded. They are recreated when needed
        after deserialization.
        """
        nrt = False
        if self._linking_files:
            if (
                len(self._linking_files) == 1
                and NRT_LIBRARY in self._linking_files
            ):
                nrt = True
            else:
                msg = "Cannot pickle CUDACodeLibrary with linking files"
                raise RuntimeError(msg)

        if not self._finalized:
            raise RuntimeError("Cannot pickle unfinalized CUDACodeLibrary")
        return dict(
            codegen=None,
            name=self.name,
            entry_name=self._entry_name,
            llvm_strs=self.llvm_strs,
            ptx_cache=self._ptx_cache,
            cubin_cache=self._cubin_cache,
            linkerinfo_cache=self._linkerinfo_cache,
            max_registers=self._max_registers,
            nvvm_options=self._nvvm_options,
            needs_cudadevrt=self.needs_cudadevrt,
            nrt=nrt,
            use_cooperative=self.use_cooperative,
            lto=self._lto,
        )

    @classmethod
    def _rebuild(
        cls,
        codegen,
        name,
        entry_name,
        llvm_strs,
        ptx_cache,
        cubin_cache,
        linkerinfo_cache,
        max_registers,
        nvvm_options,
        needs_cudadevrt,
        nrt,
        use_cooperative,
        lto,
    ):
        """
        Rebuild an instance.
        """
        instance = cls(codegen, name, entry_name=entry_name)

        instance._llvm_strs = llvm_strs
        instance._ptx_cache = ptx_cache
        instance._cubin_cache = cubin_cache
        instance._linkerinfo_cache = linkerinfo_cache

        instance._max_registers = max_registers
        instance._nvvm_options = nvvm_options
        instance.needs_cudadevrt = needs_cudadevrt
        instance.use_cooperative = use_cooperative

        instance._finalized = True
        if nrt:
            instance._linking_files = {NRT_LIBRARY}

        instance._lto = lto
        return instance


class JITCUDACodegen(Codegen):
    """
    This codegen implementation for CUDA only generates optimized LLVM IR.
    Generation of PTX code is done separately (see numba.cuda.compiler).
    """

    _library_class = CUDACodeLibrary

    def __init__(self, module_name):
        pass

    def _create_empty_module(self, name):
        ir_module = ir.Module(name)
        ir_module.triple = CUDA_TRIPLE
        ir_module.data_layout = nvvm.NVVM().data_layout
        nvvm.add_ir_version(ir_module)
        return ir_module

    def _add_module(self, module):
        pass

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        ctx = devices.get_context()
        cc = ctx.device.compute_capability
        return (runtime.runtime.get_version(), cc)
