import ctypes
import os
from functools import wraps
import numpy as np

from numba import cuda, config
from numba.core.runtime.nrt import _nrt_mstats
from numba.cuda.cudadrv.driver import (
    _Linker,
    driver,
    launch_kernel,
    USE_NV_BINDING,
)
from numba.cuda.cudadrv import devices
from numba.cuda.api import get_current_device
from numba.cuda.utils import _readenv, cached_file_read
from numba.cuda.cudadrv.linkable_code import CUSource


# Check environment variable or config for NRT statistics enablement
NRT_STATS = _readenv("NUMBA_CUDA_NRT_STATS", bool, False) or getattr(
    config, "NUMBA_CUDA_NRT_STATS", False
)
if not hasattr(config, "NUMBA_CUDA_NRT_STATS"):
    config.CUDA_NRT_STATS = NRT_STATS


# Check environment variable or config for NRT enablement
ENABLE_NRT = _readenv("NUMBA_CUDA_ENABLE_NRT", bool, False) or getattr(
    config, "NUMBA_CUDA_ENABLE_NRT", False
)
if not hasattr(config, "NUMBA_CUDA_ENABLE_NRT"):
    config.CUDA_ENABLE_NRT = ENABLE_NRT


def get_include():
    """Return the include path for the NRT header"""
    return os.path.dirname(os.path.abspath(__file__))


# Protect method to ensure NRT memory allocation and initialization
def _alloc_init_guard(method):
    """
    Ensure NRT memory allocation and initialization before running the method
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.ensure_allocated()
        self.ensure_initialized()
        return method(self, *args, **kwargs)

    return wrapper


class _Runtime:
    """Singleton class for Numba CUDA runtime"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Runtime, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """Initialize memsys module and variable"""
        self._memsys_module = None
        self._memsys = None
        self._initialized = False

    def _compile_memsys_module(self):
        """
        Compile memsys.cu and create a module from it in the current context
        """
        # Define the path for memsys.cu
        memsys_mod = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "memsys.cu"
        )
        cc = get_current_device().compute_capability

        # Create a new linker instance and add the cu file
        linker = _Linker.new(cc=cc)
        linker.add_cu_file(memsys_mod)

        # Complete the linker and create a module from it
        cubin = linker.complete()
        ctx = devices.get_context()
        module = ctx.create_module_image(cubin)

        # Set the memsys module
        self._memsys_module = module

    def ensure_allocated(self, stream=None):
        """
        If memsys is not allocated, allocate it; otherwise, perform a no-op
        """
        if self._memsys is not None:
            return

        # Allocate the memsys
        self.allocate(stream)

    def allocate(self, stream=None):
        """
        Allocate memsys on global memory
        """
        from numba.cuda import device_array

        # Check if memsys module is defined
        if self._memsys_module is None:
            # Compile the memsys module if not defined
            self._compile_memsys_module()

        # Allocate space for NRT_MemSys
        memsys_size = ctypes.c_uint64()
        ptr, nbytes = self._memsys_module.get_global_symbol("memsys_size")
        device_memsys_size = ptr.device_ctypes_pointer
        if USE_NV_BINDING:
            device_memsys_size = device_memsys_size.value
        driver.cuMemcpyDtoH(
            ctypes.addressof(memsys_size), device_memsys_size, nbytes
        )
        self._memsys = device_array(
            (memsys_size.value,), dtype="i1", stream=stream
        )
        self.set_memsys_to_module(self._memsys_module, stream=stream)

    def _single_thread_launch(self, module, stream, name, params=()):
        """
        Launch the specified kernel with only 1 thread
        """
        if stream is None:
            stream = cuda.default_stream()

        func = module.get_function(name)
        launch_kernel(
            func.handle,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream.handle,
            params,
            cooperative=False,
        )

    def ensure_initialized(self, stream=None):
        """
        If memsys is not initialized, initialize memsys
        """
        if self._initialized:
            return

        # Initialize the memsys
        self.initialize(stream)

    def initialize(self, stream=None):
        """
        Launch memsys initialization kernel
        """
        self.ensure_allocated()

        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_init"
        )
        self._initialized = True

        if config.CUDA_NRT_STATS:
            self.memsys_enable_stats()

    @_alloc_init_guard
    def memsys_enable_stats(self, stream=None):
        """
        Enable memsys statistics
        """
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_enable_stats"
        )

    @_alloc_init_guard
    def memsys_disable_stats(self, stream=None):
        """
        Disable memsys statistics
        """
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_disable_stats"
        )

    @_alloc_init_guard
    def memsys_stats_enabled(self, stream=None):
        """
        Return a boolean indicating whether memsys is enabled. Synchronizes
        context
        """
        enabled_ar = cuda.managed_array(1, np.uint8)
        enabled_ptr = enabled_ar.device_ctypes_pointer

        self._single_thread_launch(
            self._memsys_module,
            stream,
            "NRT_MemSys_stats_enabled",
            (enabled_ptr,),
        )

        cuda.synchronize()
        return bool(enabled_ar[0])

    @_alloc_init_guard
    def _copy_memsys_to_host(self, stream):
        """
        Copy all statistics of memsys to the host
        """
        dt = np.dtype(
            [
                ("alloc", np.uint64),
                ("free", np.uint64),
                ("mi_alloc", np.uint64),
                ("mi_free", np.uint64),
            ]
        )

        stats_for_read = cuda.managed_array(1, dt)
        stats_ptr = stats_for_read.device_ctypes_pointer

        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_read", [stats_ptr]
        )
        cuda.synchronize()

        return stats_for_read[0]

    @_alloc_init_guard
    def get_allocation_stats(self, stream=None):
        """
        Get the allocation statistics
        """
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        memsys = self._copy_memsys_to_host(stream)
        return _nrt_mstats(
            alloc=memsys["alloc"],
            free=memsys["free"],
            mi_alloc=memsys["mi_alloc"],
            mi_free=memsys["mi_free"],
        )

    @_alloc_init_guard
    def _get_single_stat(self, stat, stream=None):
        """
        Get a single stat from the memsys
        """
        got = cuda.managed_array(1, np.uint64)
        got_ptr = got.device_ctypes_pointer

        self._single_thread_launch(
            self._memsys_module, stream, f"NRT_MemSys_read_{stat}", [got_ptr]
        )

        cuda.synchronize()
        return got[0]

    @_alloc_init_guard
    def memsys_get_stats_alloc(self, stream=None):
        """
        Get the allocation statistic
        """
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")

        return self._get_single_stat("alloc")

    @_alloc_init_guard
    def memsys_get_stats_free(self, stream=None):
        """
        Get the free statistic
        """
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")

        return self._get_single_stat("free")

    @_alloc_init_guard
    def memsys_get_stats_mi_alloc(self, stream=None):
        """
        Get the mi alloc statistic
        """
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")

        return self._get_single_stat("mi_alloc")

    @_alloc_init_guard
    def memsys_get_stats_mi_free(self, stream=None):
        """
        Get the mi free statistic
        """
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")

        return self._get_single_stat("mi_free")

    def set_memsys_to_module(self, module, stream=None):
        """
        Set the memsys module. The module must contain `NRT_MemSys_set` kernel,
        and declare a pointer to NRT_MemSys structure.
        """
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before setting to module."
            )

        memsys_ptr = self._memsys.device_ctypes_pointer

        self._single_thread_launch(
            module, stream, "NRT_MemSys_set", [memsys_ptr]
        )

    @_alloc_init_guard
    def print_memsys(self, stream=None):
        """
        Print the current statistics of memsys, for debugging purposes
        """
        cuda.synchronize()
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_print"
        )


# Create an instance of the runtime
rtsys = _Runtime()


basedir = os.path.dirname(os.path.abspath(__file__))
nrt_path = os.path.join(basedir, "nrt.cu")
nrt_src = cached_file_read(nrt_path)
NRT_LIBRARY = CUSource(nrt_src, name="nrt.cu", nrt=True)
