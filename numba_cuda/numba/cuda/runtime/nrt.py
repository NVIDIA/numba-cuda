import os
from functools import wraps
import numpy as np

from numba import cuda, config
from numba.core.runtime.nrt import _nrt_mstats
from numba.cuda.cudadrv.driver import Linker, launch_kernel
from numba.cuda.cudadrv import devices
from numba.cuda.api import get_current_device
from numba.cuda.utils import _readenv


NRT_STATS = (
    _readenv("NUMBA_CUDA_NRT_STATS", bool, False) or
    getattr(config, "NUMBA_CUDA_NRT_STATS", False)
)
if not hasattr(config, "NUMBA_CUDA_NRT_STATS"):
    config.CUDA_NRT_STATS = NRT_STATS

ENABLE_NRT = (
    _readenv("NUMBA_CUDA_ENABLE_NRT", bool, False) or
    getattr(config, "NUMBA_CUDA_ENABLE_NRT", False)
)
if not hasattr(config, "NUMBA_CUDA_ENABLE_NRT"):
    config.CUDA_ENABLE_NRT = ENABLE_NRT


def _alloc_init_guard(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.ensure_allocated()
        self.ensure_initialized()
        return method(self, *args, **kwargs)
    return wrapper


class _Runtime:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Runtime, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self._memsys_module = None
        self._memsys = None

        self._initialized = False

    def _compile_memsys_module(self):
        memsys_mod = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "memsys.cu"
        )
        cc = get_current_device().compute_capability

        linker = Linker.new(cc=cc)
        linker.add_cu_file(memsys_mod)
        cubin = linker.complete()

        ctx = devices.get_context()
        module = ctx.create_module_image(cubin)

        self._memsys_module = module

    def ensure_allocated(self, stream=None):
        if self._memsys is not None:
            return

        self.allocate(stream)

    def allocate(self, stream=None):
        from numba.cuda import device_array

        if self._memsys_module is None:
            self._compile_memsys_module()

        # Allocate space for NRT_MemSys
        # TODO: determine the size of NRT_MemSys at runtime
        self._memsys = device_array((40,), dtype="i1", stream=stream)
        # TODO: Memsys module needs a stream that's consistent with the
        # system's stream.
        self.set_memsys_to_module(self._memsys_module, stream=stream)

    def _single_thread_launch(self, module, stream, name, params=()):
        if stream is None:
            stream = cuda.default_stream()

        func = module.get_function(name)
        launch_kernel(
            func.handle,
            1, 1, 1,
            1, 1, 1,
            0,
            stream.handle,
            params,
            cooperative=False
        )

    def ensure_initialized(self, stream=None):
        if self._initialized:
            return

        self.initialize(stream)

    def initialize(self, stream=None):
        self.ensure_allocated(stream)

        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_init")
        self._initialized = True

        if NRT_STATS:
            self.memsys_enable_stats(stream)

    @_alloc_init_guard
    def memsys_enable_stats(self, stream=None):
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_enable_stats")

    @_alloc_init_guard
    def memsys_disable_stats(self, stream=None):
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_disable_stats")

    @_alloc_init_guard
    def memsys_stats_enabled(self, stream=None):
        enabled_ar = cuda.managed_array(1, np.uint8)

        self._single_thread_launch(
            self._memsys_module,
            stream,
            "NRT_MemSys_stats_get_enabled",
            (enabled_ar.device_ctypes_pointer,)
        )

        cuda.synchronize()
        return bool(enabled_ar[0])

    @_alloc_init_guard
    def _copy_memsys_to_host(self, stream):

        # Q: What stream should we execute this on?
        dt = np.dtype([
            ('alloc', np.uint64),
            ('free', np.uint64),
            ('mi_alloc', np.uint64),
            ('mi_free', np.uint64)
        ])

        stats_for_read = cuda.managed_array(1, dt)

        self._single_thread_launch(
            self._memsys_module,
            stream,
            "NRT_MemSys_read",
            [stats_for_read.device_ctypes_pointer]
        )
        cuda.synchronize()

        return stats_for_read[0]

    @_alloc_init_guard
    def get_allocation_stats(self, stream=None):
        memsys = self._copy_memsys_to_host(stream)
        return _nrt_mstats(
            alloc=memsys["alloc"],
            free=memsys["free"],
            mi_alloc=memsys["mi_alloc"],
            mi_free=memsys["mi_free"]
        )

    def set_memsys_to_module(self, module, stream=None):
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before initializing.")

        print(f"Setting {self._memsys.device_ctypes_pointer} to {module}")
        self._single_thread_launch(
            module,
            stream,
            "NRT_MemSys_set",
            [self._memsys.device_ctypes_pointer,]
        )

    @_alloc_init_guard
    def print_memsys(self, stream=None):
        cuda.synchronize()
        self._single_thread_launch(
            self._memsys_module,
            stream,
            "NRT_MemSys_print"
        )


rtsys = _Runtime()
