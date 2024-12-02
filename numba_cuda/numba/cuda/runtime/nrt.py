import os
import numpy as np

from numba import cuda
from numba.core.runtime.nrt import _nrt_mstats
from numba.cuda.cudadrv.driver import Linker, launch_kernel
from numba.cuda.cudadrv import devices
from numba.cuda.api import get_current_device


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

    def _ensure_allocate(self):
        if self._memsys is not None:
            return

        self.allocate()

    def allocate(self):
        from numba.cuda import device_array

        if self._memsys_module is None:
            self._compile_memsys_module()

        if self._memsys is None:
            # Allocate space for NRT_MemSys
            # TODO: determine the size of NRT_MemSys at runtime
            self._memsys = device_array((40,), dtype="i1")

    def _single_thread_launch(self, module, stream, name, params=()):
        func = module.get_function(name)
        launch_kernel(
            func.handle,
            1, 1, 1,
            1, 1, 1,
            0,
            stream,
            params,
            cooperative=False
        )

    def _ensure_initialize(self, stream):
        if self._initialized:
            return

        self.initialize(stream)

    def initialize(self, stream):
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before initializing.")

        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_init")
        self._initialized = True

    def enable(self, stream):
        self._single_thread_launch(
            self._memsys_module, stream, "NR_MemSys_enable")

    def disable(self, stream):
        self._single_thread_launch(
            self._memsys_module, stream, "NR_MemSys_disable")

    def _copy_memsys_to_host(self, stream=0):
        self._ensure_allocate()
        self._ensure_initialize(stream)

        # Q: What stream should we execute this on?
        # read the stats
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

    def get_allocation_stats(self):
        memsys = self._copy_memsys_to_host()
        return _nrt_mstats(
            alloc=memsys.alloc,
            free=memsys.free,
            mi_alloc=memsys.mi_alloc,
            mi_free=memsys.mi_free
        )

    def set_memsys_to_module(self, module, stream):
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before initializing.")

        self._single_thread_launch(
            module,
            stream,
            "NRT_MemSys_set",
            [self._memsys.device_ctypes_pointer,]
        )


rtsys = _Runtime()
