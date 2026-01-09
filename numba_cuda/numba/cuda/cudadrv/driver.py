# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
CUDA driver bridge implementation

NOTE:
The new driver implementation uses a *_PendingDeallocs* that help prevents a
crashing the system (particularly OSX) when the CUDA context is corrupted at
resource deallocation.  The old approach ties resource management directly
into the object destructor; thus, at corruption of the CUDA context,
subsequent deallocation could further corrupt the CUDA context and causes the
system to freeze in some cases.

"""

import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
import subprocess
import tempfile
import re
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (
    c_int,
    byref,
    c_size_t,
    c_void_p,
    c_uint8,
)
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque


from numba.cuda.cext import mviewbuf
from numba.cuda.core import config
from numba.cuda import utils, serialize
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj
from .mappings import FILE_EXTENSION_MAP
from .linkable_code import LinkableCode, LTOIR, Fatbin, Object
from numba.cuda.utils import cached_file_read
from numba.cuda.cudadrv import enums, drvapi, nvrtc

from cuda.bindings import driver as binding
from numba.cuda._compat import (
    Linker,
    LinkerOptions,
    ObjectCode,
    Stream as ExperimentalStream,
    Device as ExperimentalDevice,
)

from cuda.bindings.utils import get_cuda_native_handle


# There is no definition of the default stream in the Nvidia bindings (nor
# is there at the C/C++ level), so we define it here so we don't need to
# use a magic number 0 in places where we want the default stream.
CU_STREAM_DEFAULT = 0


MIN_REQUIRED_CC = (3, 5)
SUPPORTS_IPC = sys.platform.startswith("linux")


_py_decref = ctypes.pythonapi.Py_DecRef
_py_incref = ctypes.pythonapi.Py_IncRef
_py_decref.argtypes = [ctypes.py_object]
_py_incref.argtypes = [ctypes.py_object]


def make_logger():
    logger = logging.getLogger(__name__)
    # is logging configured?
    if not logger.hasHandlers():
        # read user config
        lvl = str(config.CUDA_LOG_LEVEL).upper()
        lvl = getattr(logging, lvl, None)
        if not isinstance(lvl, int):
            # default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)
        # did user specify a level?
        if config.CUDA_LOG_LEVEL:
            # create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = "== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s"
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # otherwise, put a null handler
            logger.addHandler(logging.NullHandler())
    return logger


@functools.cache
def _have_nvjitlink():
    try:
        from cuda.bindings._internal import nvjitlink as nvjitlink_internal
        from cuda.bindings._internal.utils import NotSupportedError
    except ImportError:
        return False

    try:
        if (
            nvjitlink_internal._inspect_function_pointer("__nvJitLinkVersion")
            == 0
        ):
            return False
        try:
            from cuda.bindings import nvjitlink

            if nvjitlink.version() < (12, 3):
                return False
        except Exception:
            return False
        return True
    except (RuntimeError, NotSupportedError):
        return False


class DeadMemoryError(RuntimeError):
    pass


class LinkerError(RuntimeError):
    pass


class CudaAPIError(CudaDriverError):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super(CudaAPIError, self).__init__(code, msg)

    def __str__(self):
        return "[%s] %s" % (self.code, self.msg)


def locate_driver_and_loader():
    # Determine DLL type
    if sys.platform == "win32":
        dlloader = ctypes.WinDLL
        dldir = ["\\windows\\system32"]
        dlnames = ["nvcuda.dll"]
    elif sys.platform == "darwin":
        dlloader = ctypes.CDLL
        dldir = ["/usr/local/cuda/lib"]
        dlnames = ["libcuda.dylib"]
    else:
        # Assume to be *nix like
        dlloader = ctypes.CDLL
        dldir = ["/usr/lib", "/usr/lib64"]
        dlnames = ["libcuda.so", "libcuda.so.1"]

    # First search for the name in the default library path.
    # If that is not found, try specific common paths.
    candidates = dlnames + [
        os.path.join(x, y) for x, y in product(dldir, dlnames)
    ]

    return dlloader, candidates


def load_driver(dlloader, candidates):
    # Load the driver; Collect driver error information
    path_not_exist = []
    driver_load_error = []

    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:  # noqa: PERF203
            # Problem opening the DLL
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return dll, path

    # Problem loading driver
    if all(path_not_exist):
        _raise_driver_not_found()
    else:
        errmsg = "\n".join(str(e) for e in driver_load_error)
        _raise_driver_error(errmsg)


def find_driver():
    dlloader, candidates = locate_driver_and_loader()
    dll, path = load_driver(dlloader, candidates)
    return dll


DRIVER_NOT_FOUND_MSG = """
CUDA driver library cannot be found.
Ensure that a compatible NVIDIA driver is installed and available on your system path.
"""

DRIVER_LOAD_ERROR_MSG = """
Possible CUDA driver libraries are found but error occurred during load:
%s
"""


def _raise_driver_not_found():
    raise CudaSupportError(DRIVER_NOT_FOUND_MSG)


def _raise_driver_error(e):
    raise CudaSupportError(DRIVER_LOAD_ERROR_MSG % e)


def _build_reverse_error_map():
    prefix = "CUDA_ERROR"
    map = utils.UniqueDict()
    for name in dir(enums):
        if name.startswith(prefix):
            code = getattr(enums, name)
            map[code] = name
    return map


def _getpid():
    return os.getpid()


ERROR_MAP = _build_reverse_error_map()


class Driver(object):
    """
    Driver API functions are lazily bound.
    """

    _singleton = None

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            cls._singleton = obj
        return obj

    def __init__(self):
        self.devices = utils.UniqueDict()
        self.is_initialized = False
        self.initialization_error = None
        self.pid = None
        try:
            if config.DISABLE_CUDA:
                msg = (
                    "CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 "
                    "in the environment, or because CUDA is unsupported on "
                    "32-bit systems."
                )
                raise CudaSupportError(msg)
            self.lib = find_driver()
        except CudaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e.msg

    def ensure_initialized(self):
        if self.is_initialized:
            return

        # lazily initialize logger
        global _logger
        _logger = make_logger()

        self.is_initialized = True
        try:
            _logger.info("init")
            self.cuInit(0)
        except CudaAPIError as e:
            description = f"{e.msg} ({e.code})"
            self.initialization_error = description
            raise CudaSupportError(f"Error at driver init: {description}")
        else:
            self.pid = _getpid()

    @property
    def is_available(self):
        self.ensure_initialized()
        return self.initialization_error is None

    def __getattr__(self, fname):
        # First request of a driver API function
        self.ensure_initialized()

        if self.initialization_error is not None:
            raise CudaSupportError(
                "Error at driver init: \n%s:" % self.initialization_error
            )

        return self._cuda_python_wrap_fn(fname)

    def _ctypes_wrap_fn(self, fname, libfn=None):
        # Wrap a CUDA driver function by default
        if libfn is None:
            try:
                proto = API_PROTOTYPES[fname]
            except KeyError:
                raise AttributeError(fname)
            restype = proto[0]
            argtypes = proto[1:]

            # Find function in driver library
            libfn = self._find_api(fname)
            libfn.restype = restype
            libfn.argtypes = argtypes

        def verbose_cuda_api_call(*args):
            argstr = ", ".join([str(arg) for arg in args])
            _logger.debug("call driver api: %s(%s)", libfn.__name__, argstr)
            retcode = libfn(*args)
            self._check_ctypes_error(fname, retcode)

        def safe_cuda_api_call(*args):
            _logger.debug("call driver api: %s", libfn.__name__)
            retcode = libfn(*args)
            self._check_ctypes_error(fname, retcode)

        if config.CUDA_LOG_API_ARGS:
            wrapper = verbose_cuda_api_call
        else:
            wrapper = safe_cuda_api_call

        safe_call = functools.wraps(libfn)(wrapper)
        setattr(self, fname, safe_call)
        return safe_call

    def _cuda_python_wrap_fn(self, fname):
        libfn = getattr(binding, fname)

        def verbose_cuda_api_call(*args):
            argstr = ", ".join([str(arg) for arg in args])
            _logger.debug("call driver api: %s(%s)", libfn.__name__, argstr)
            return self._check_cuda_python_error(fname, libfn(*args))

        def safe_cuda_api_call(*args):
            _logger.debug("call driver api: %s", libfn.__name__)
            return self._check_cuda_python_error(fname, libfn(*args))

        if config.CUDA_LOG_API_ARGS:
            wrapper = verbose_cuda_api_call
        else:
            wrapper = safe_cuda_api_call

        safe_call = functools.wraps(libfn)(wrapper)
        setattr(self, fname, safe_call)
        return safe_call

    def _find_api(self, fname):
        # We use alternatively-named functions for PTDS with the Numba ctypes
        # binding. It handles linking to the correct variant.
        variants = ("_v2", "")

        if fname in ("cuCtxGetDevice", "cuCtxSynchronize"):
            return getattr(self.lib, fname)

        for variant in variants:
            if (
                value := getattr(self.lib, f"{fname}{variant}", None)
            ) is not None:
                return value

        # Not found.
        # Delay missing function error to use
        def absent_function(*args, **kws):
            raise CudaDriverError(f"Driver missing function: {fname}")

        setattr(self, fname, absent_function)
        return absent_function

    def _detect_fork(self):
        if self.pid is not None and _getpid() != self.pid:
            msg = "pid %s forked from pid %s after CUDA driver init"
            _logger.critical(msg, _getpid(), self.pid)
            raise CudaDriverError("CUDA initialized before forking")

    def _check_ctypes_error(self, fname, retcode):
        if retcode != enums.CUDA_SUCCESS:
            errname = ERROR_MAP.get(retcode, "UNKNOWN_CUDA_ERROR")
            msg = "Call to %s results in %s" % (fname, errname)
            _logger.error(msg)
            if retcode == enums.CUDA_ERROR_NOT_INITIALIZED:
                self._detect_fork()
            raise CudaAPIError(retcode, msg)

    def _check_cuda_python_error(self, fname, returned):
        retcode = returned[0]
        retval = returned[1:]
        if len(retval) == 1:
            retval = retval[0]

        if retcode != binding.CUresult.CUDA_SUCCESS:
            msg = "Call to %s results in %s" % (fname, retcode.name)
            _logger.error(msg)
            if retcode == binding.CUresult.CUDA_ERROR_NOT_INITIALIZED:
                self._detect_fork()
            raise CudaAPIError(retcode, msg)

        return retval

    def get_device(self, devnum=0):
        dev = self.devices.get(devnum)
        if dev is None:
            dev = Device(devnum)
            self.devices[devnum] = dev
        return weakref.proxy(dev)

    def get_device_count(self):
        return self.cuDeviceGetCount()

    def list_devices(self):
        """Returns a list of active devices"""
        return list(self.devices.values())

    def reset(self):
        """Reset all devices"""
        for dev in self.devices.values():
            dev.reset()

    def pop_active_context(self):
        """Pop the active CUDA context and return the handle.
        If no CUDA context is active, return None.
        """
        with self.get_active_context() as ac:
            if ac.devnum is not None:
                popped = drvapi.cu_context(int(driver.cuCtxPopCurrent()))
                return popped

    def get_active_context(self):
        """Returns an instance of ``_ActiveContext``."""
        return _ActiveContext()

    def get_version(self):
        """
        Returns the CUDA Driver version as a tuple (major, minor).
        """
        # The version is encoded as (1000 * major) + (10 * minor)
        version = driver.cuDriverGetVersion()
        major = version // 1000
        minor = (version - (major * 1000)) // 10
        return (major, minor)


class _ActiveContext(object):
    """An contextmanager object to cache active context to reduce dependency
    on querying the CUDA driver API.

    Once entering the context, it is assumed that the active CUDA context is
    not changed until the context is exited.
    """

    _tls_cache = threading.local()

    def __enter__(self):
        is_top = False
        # check TLS cache
        cache = self._tls_cache
        try:
            hctx, devnum = cache.ctx_devnum
        except AttributeError:
            # Not cached. Query the driver API.
            hctx = driver.cuCtxGetCurrent()
            if int(hctx) == 0:
                hctx = None
            else:
                hctx = drvapi.cu_context(int(hctx))

            if hctx is None:
                devnum = None
            else:
                devnum = int(driver.cuCtxGetDevice())

                self._tls_cache.ctx_devnum = (hctx, devnum)
                is_top = True

        self._is_top = is_top
        self.context_handle = hctx
        self.devnum = devnum
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_top:
            del self._tls_cache.ctx_devnum

    def __bool__(self):
        """Returns True is there's a valid and active CUDA context."""
        return self.context_handle is not None

    __nonzero__ = __bool__


driver = Driver()


def _build_reverse_device_attrs():
    prefix = "CU_DEVICE_ATTRIBUTE_"
    map = utils.UniqueDict()
    for name in dir(enums):
        if name.startswith(prefix):
            map[name[len(prefix) :]] = getattr(enums, name)
    return map


DEVICE_ATTRIBUTES = _build_reverse_device_attrs()


class Device:
    """
    The device object owns the CUDA contexts.  This is owned by the driver
    object.  User should not construct devices directly.
    """

    @classmethod
    def from_identity(self, identity):
        """Create Device object from device identity created by
        ``Device.get_device_identity()``.
        """
        for devid in range(driver.get_device_count()):
            d = driver.get_device(devid)
            if d.get_device_identity() == identity:
                return d
        else:
            raise RuntimeError(
                f"No device of {identity} is found. "
                "Target device may not be visible in this process."
            )

    def __init__(self, devnum: int) -> None:
        self._dev = ExperimentalDevice(devnum)
        self.id = self._dev.device_id
        self.compute_capability = self._dev.compute_capability
        self.name = self._dev.name
        self.uuid = f"GPU-{self._dev.uuid}"
        self.primary_context = None

    def get_device_identity(self):
        return {
            "pci_domain_id": self.PCI_DOMAIN_ID,
            "pci_bus_id": self.PCI_BUS_ID,
            "pci_device_id": self.PCI_DEVICE_ID,
        }

    def __repr__(self):
        return f"<CUDA device {self.id:d} '{self.name}'>"

    def __getattr__(self, attr):
        """Read attributes lazily"""
        code = getattr(
            binding.CUdevice_attribute, f"CU_DEVICE_ATTRIBUTE_{attr}"
        )
        value = driver.cuDeviceGetAttribute(code, self.id)

        setattr(self, attr, value)
        return value

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Device) and self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    def get_primary_context(self):
        """
        Returns the primary context for the device.
        Note: it is not pushed to the CPU thread.
        """
        if (ctx := self.primary_context) is not None:
            return ctx

        if self.compute_capability < MIN_REQUIRED_CC:
            raise CudaSupportError(
                f"{self} has compute capability < {MIN_REQUIRED_CC}"
            )

        self._dev.set_current()
        self.primary_context = ctx = Context(
            weakref.proxy(self),
            ctypes.c_void_p(int(self._dev.context._handle)),
        )
        return ctx

    def release_primary_context(self):
        """
        Release reference to primary context if it has been retained.
        """
        if self.primary_context:
            driver.cuDevicePrimaryCtxRelease(self.id)
            self.primary_context = None

    def reset(self):
        try:
            if (ctx := self.primary_context) is not None:
                ctx.reset()
            self.release_primary_context()
        finally:
            # reset at the driver level
            driver.cuDevicePrimaryCtxReset(self.id)

    @property
    def supports_float16(self):
        return self.compute_capability >= (5, 3)

    @property
    def supports_bfloat16(self):
        return self.compute_capability >= (8, 0)


class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
    """Abstract base class for External Memory Management (EMM) Plugins."""

    def __init__(self, *args, **kwargs):
        if "context" not in kwargs:
            raise RuntimeError("Memory manager requires a context")
        self.context = kwargs.pop("context")

    @abstractmethod
    def memalloc(self, size):
        """
        Allocate on-device memory in the current context.

        :param size: Size of allocation in bytes
        :type size: int
        :return: A memory pointer instance that owns the allocated memory
        :rtype: :class:`MemoryPointer`
        """

    @abstractmethod
    def memhostalloc(self, size, mapped, portable, wc):
        """
        Allocate pinned host memory.

        :param size: Size of the allocation in bytes
        :type size: int
        :param mapped: Whether the allocated memory should be mapped into the
                       CUDA address space.
        :type mapped: bool
        :param portable: Whether the memory will be considered pinned by all
                         contexts, and not just the calling context.
        :type portable: bool
        :param wc: Whether to allocate the memory as write-combined.
        :type wc: bool
        :return: A memory pointer instance that owns the allocated memory. The
                 return type depends on whether the region was mapped into
                 device memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """

    @abstractmethod
    def mempin(self, owner, pointer, size, mapped):
        """
        Pin a region of host memory that is already allocated.

        :param owner: The object that owns the memory.
        :param pointer: The pointer to the beginning of the region to pin.
        :type pointer: int
        :param size: The size of the region in bytes.
        :type size: int
        :param mapped: Whether the region should also be mapped into device
                       memory.
        :type mapped: bool
        :return: A memory pointer instance that refers to the allocated
                 memory.
        :rtype: :class:`MappedMemory` or :class:`PinnedMemory`
        """

    @abstractmethod
    def initialize(self):
        """
        Perform any initialization required for the EMM plugin instance to be
        ready to use.

        :return: None
        """

    @abstractmethod
    def get_ipc_handle(self, memory):
        """
        Return an IPC handle from a GPU allocation.

        :param memory: Memory for which the IPC handle should be created.
        :type memory: :class:`MemoryPointer`
        :return: IPC handle for the allocation
        :rtype: :class:`IpcHandle`
        """

    @abstractmethod
    def get_memory_info(self):
        """
        Returns ``(free, total)`` memory in bytes in the context. May raise
        :class:`NotImplementedError`, if returning such information is not
        practical (e.g. for a pool allocator).

        :return: Memory info
        :rtype: :class:`MemoryInfo`
        """

    @abstractmethod
    def reset(self):
        """
        Clears up all memory allocated in this context.

        :return: None
        """

    @abstractmethod
    def defer_cleanup(self):
        """
        Returns a context manager that ensures the implementation of deferred
        cleanup whilst it is active.

        :return: Context manager
        """

    @property
    @abstractmethod
    def interface_version(self):
        """
        Returns an integer specifying the version of the EMM Plugin interface
        supported by the plugin implementation. Should always return 1 for
        implementations of this version of the specification.
        """


class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):
    """Base class for External Memory Management (EMM) Plugins that only
    implement on-device allocation. A subclass need not implement the
    ``memhostalloc`` and ``mempin`` methods.

    This class also implements ``reset`` and ``defer_cleanup`` (see
    :class:`numba.cuda.BaseCUDAMemoryManager`) for its own internal state
    management. If an EMM Plugin based on this class also implements these
    methods, then its implementations of these must also call the method from
    ``super()`` to give ``HostOnlyCUDAMemoryManager`` an opportunity to do the
    necessary work for the host allocations it is managing.

    This class does not implement ``interface_version``, as it will always be
    consistent with the version of Numba in which it is implemented. An EMM
    Plugin subclassing this class should implement ``interface_version``
    instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocations = utils.UniqueDict()
        self.deallocations = _PendingDeallocs()

    def _attempt_allocation(self, allocator):
        """
        Attempt allocation by calling *allocator*.  If an out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
        try:
            return allocator()
        except CudaAPIError as e:
            # is out-of-memory?
            oom_code = binding.CUresult.CUDA_ERROR_OUT_OF_MEMORY
            if e.code == oom_code:
                # clear pending deallocations
                self.deallocations.clear()
                # try again
                return allocator()
            else:
                raise

    def memhostalloc(self, size, mapped=False, portable=False, wc=False):
        """Implements the allocation of pinned host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= enums.CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= enums.CU_MEMHOSTALLOC_WRITECOMBINED

        def allocator():
            return driver.cuMemHostAlloc(size, flags)

        if mapped:
            pointer = self._attempt_allocation(allocator)
        else:
            pointer = allocator()

        alloc_key = pointer

        finalizer = _hostalloc_finalizer(self, pointer, alloc_key, size, mapped)
        ctx = weakref.proxy(self.context)

        if mapped:
            mem = MappedMemory(ctx, pointer, size, finalizer=finalizer)
            self.allocations[alloc_key] = mem
            return mem.own()
        else:
            return PinnedMemory(ctx, pointer, size, finalizer=finalizer)

    def mempin(self, owner, pointer, size, mapped=False):
        """Implements the pinning of host memory.

        It is recommended that this method is not overridden by EMM Plugin
        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
        """
        alloc_key = pointer

        # possible flags are "portable" (between context)
        # and "device-map" (map host memory to device thus no need
        # for memory transfer).
        flags = 0

        if mapped:
            flags |= enums.CU_MEMHOSTREGISTER_DEVICEMAP

        def allocator():
            driver.cuMemHostRegister(pointer, size, flags)

        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()

        finalizer = _pin_finalizer(self, pointer, alloc_key, mapped)
        ctx = weakref.proxy(self.context)

        if mapped:
            mem = MappedMemory(
                ctx, pointer, size, owner=owner, finalizer=finalizer
            )
            self.allocations[alloc_key] = mem
            return mem.own()
        else:
            return PinnedMemory(
                ctx, pointer, size, owner=owner, finalizer=finalizer
            )

    def memallocmanaged(self, size, attach_global):
        def allocator():
            ma_flags = binding.CUmemAttach_flags

            if attach_global:
                flags = ma_flags.CU_MEM_ATTACH_GLOBAL.value
            else:
                flags = ma_flags.CU_MEM_ATTACH_HOST.value

            return driver.cuMemAllocManaged(size, flags)

        ptr = self._attempt_allocation(allocator)

        alloc_key = ptr

        finalizer = _alloc_finalizer(self, ptr, alloc_key, size)
        ctx = weakref.proxy(self.context)
        mem = ManagedMemory(ctx, ptr, size, finalizer=finalizer)
        self.allocations[alloc_key] = mem
        return mem.own()

    def reset(self):
        """Clears up all host memory (mapped and/or pinned) in the current
        context.

        EMM Plugins that override this method must call ``super().reset()`` to
        ensure that host allocations are also cleaned up."""
        self.allocations.clear()
        self.deallocations.clear()

    @contextlib.contextmanager
    def defer_cleanup(self):
        """Returns a context manager that disables cleanup of mapped or pinned
        host memory in the current context whilst it is active.

        EMM Plugins that override this method must obtain the context manager
        from this method before yielding to ensure that cleanup of host
        allocations is also deferred."""
        with self.deallocations.disable():
            yield


class GetIpcHandleMixin:
    """A class that provides a default implementation of ``get_ipc_handle()``."""

    def get_ipc_handle(self, memory):
        """Open an IPC memory handle by using ``cuMemGetAddressRange`` to
        determine the base pointer of the allocation. An IPC handle of type
        ``cu_ipc_mem_handle`` is constructed and initialized with
        ``cuIpcGetMemHandle``. A :class:`numba.cuda.IpcHandle` is returned,
        populated with the underlying ``ipc_mem_handle``.
        """
        base, end = device_extents(memory)
        ipchandle = driver.cuIpcGetMemHandle(base)
        offset = int(memory.handle) - int(base)
        source_info = self.context.device.get_device_identity()

        return IpcHandle(
            memory, ipchandle, memory.size, source_info, offset=offset
        )


class NumbaCUDAMemoryManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    """Internal on-device memory management for Numba. This is implemented using
    the EMM Plugin interface, but is not part of the public API."""

    def initialize(self):
        # Set the memory capacity of *deallocations* as the memory manager
        # becomes active for the first time
        if self.deallocations.memory_capacity == _SizeNotSet:
            self.deallocations.memory_capacity = self.get_memory_info().total

    def memalloc(self, size):
        def allocator():
            return driver.cuMemAlloc(size)

        ptr = self._attempt_allocation(allocator)
        alloc_key = ptr

        finalizer = _alloc_finalizer(self, ptr, alloc_key, size)
        ctx = weakref.proxy(self.context)
        mem = AutoFreePointer(ctx, ptr, size, finalizer=finalizer)
        self.allocations[alloc_key] = mem
        return mem.own()

    def get_memory_info(self):
        free, total = driver.cuMemGetInfo()
        return MemoryInfo(free=free, total=total)

    @property
    def interface_version(self):
        return _SUPPORTED_EMM_INTERFACE_VERSION


_SUPPORTED_EMM_INTERFACE_VERSION = 1

_memory_manager = None


def _ensure_memory_manager():
    global _memory_manager

    if _memory_manager:
        return

    if config.CUDA_MEMORY_MANAGER == "default":
        _memory_manager = NumbaCUDAMemoryManager
        return

    try:
        mgr_module = importlib.import_module(config.CUDA_MEMORY_MANAGER)
        set_memory_manager(mgr_module._numba_memory_manager)
    except Exception:
        raise RuntimeError(
            "Failed to use memory manager from %s" % config.CUDA_MEMORY_MANAGER
        )


def set_memory_manager(mm_plugin):
    """Configure Numba to use an External Memory Management (EMM) Plugin. If
    the EMM Plugin version does not match one supported by this version of
    Numba, a RuntimeError will be raised.

    :param mm_plugin: The class implementing the EMM Plugin.
    :type mm_plugin: BaseCUDAMemoryManager
    :return: None
    """
    global _memory_manager

    dummy = mm_plugin(context=None)
    iv = dummy.interface_version
    if iv != _SUPPORTED_EMM_INTERFACE_VERSION:
        err = "EMM Plugin interface has version %d - version %d required" % (
            iv,
            _SUPPORTED_EMM_INTERFACE_VERSION,
        )
        raise RuntimeError(err)

    _memory_manager = mm_plugin


class _SizeNotSet(int):
    """
    Dummy object for _PendingDeallocs when *size* is not set.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, 0)

    def __str__(self):
        return "?"


_SizeNotSet = _SizeNotSet()


class _PendingDeallocs(object):
    """
    Pending deallocations of a context (or device since we are using the primary
    context). The capacity defaults to being unset (_SizeNotSet) but can be
    modified later once the driver is initialized and the total memory capacity
    known.
    """

    def __init__(self, capacity=_SizeNotSet):
        self._cons = deque()
        self._disable_count = 0
        self._size = 0
        self.memory_capacity = capacity

    @property
    def _max_pending_bytes(self):
        return int(self.memory_capacity * config.CUDA_DEALLOCS_RATIO)

    def add_item(self, dtor, handle, size=_SizeNotSet):
        """
        Add a pending deallocation.

        The *dtor* arg is the destructor function that takes an argument,
        *handle*.  It is used as ``dtor(handle)``.  The *size* arg is the
        byte size of the resource added.  It is an optional argument.  Some
        resources (e.g. CUModule) has an unknown memory footprint on the device.
        """
        _logger.info("add pending dealloc: %s %s bytes", dtor.__name__, size)
        self._cons.append((dtor, handle, size))
        self._size += int(size)
        if (
            len(self._cons) > config.CUDA_DEALLOCS_COUNT
            or self._size > self._max_pending_bytes
        ):
            self.clear()

    def clear(self):
        """
        Flush any pending deallocations unless it is disabled.
        Do nothing if disabled.
        """
        if not self.is_disabled:
            while self._cons:
                [dtor, handle, size] = self._cons.popleft()
                _logger.info("dealloc: %s %s bytes", dtor.__name__, size)
                dtor(handle)

            self._size = 0

    @contextlib.contextmanager
    def disable(self):
        """
        Context manager to temporarily disable flushing pending deallocation.
        This can be nested.
        """
        self._disable_count += 1
        try:
            yield
        finally:
            self._disable_count -= 1
            assert self._disable_count >= 0

    @property
    def is_disabled(self):
        return self._disable_count > 0

    def __len__(self):
        """
        Returns number of pending deallocations.
        """
        return len(self._cons)


MemoryInfo = namedtuple("MemoryInfo", "free,total")
"""Free and total memory for a device.

.. py:attribute:: free

   Free device memory in bytes.

.. py:attribute:: total

    Total device memory in bytes.
"""


class Context(object):
    """
    This object wraps a CUDA Context resource.

    Contexts should not be constructed directly by user code.
    """

    def __init__(self, device, handle):
        self.device = device
        self.handle = handle
        self.allocations = utils.UniqueDict()
        self.deallocations = _PendingDeallocs()
        _ensure_memory_manager()
        self.memory_manager = _memory_manager(context=self)
        self.modules = utils.UniqueDict()
        # For storing context specific data
        self.extras = {}

    def reset(self):
        """
        Clean up all owned resources in this context.
        """
        # Free owned resources
        _logger.info("reset context of device %s", self.device.id)
        self.memory_manager.reset()
        self.modules.clear()
        # Clear trash
        self.deallocations.clear()

    def get_memory_info(self):
        """Returns (free, total) memory in bytes in the context."""
        return self.memory_manager.get_memory_info()

    def get_active_blocks_per_multiprocessor(
        self, func, blocksize, memsize, flags=None
    ):
        """Return occupancy of a function.
        :param func: kernel for which occupancy is calculated
        :param blocksize: block size the kernel is intended to be launched with
        :param memsize: per-block dynamic shared memory usage intended, in bytes
        """
        args = (func, blocksize, memsize, flags)
        return self._cuda_python_active_blocks_per_multiprocessor(*args)

    def _cuda_python_active_blocks_per_multiprocessor(
        self, func, blocksize, memsize, flags
    ):
        ps = [func.handle, blocksize, memsize]

        if not flags:
            return driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(*ps)

        ps.append(flags)
        return driver.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(*ps)

    def _ctypes_active_blocks_per_multiprocessor(
        self, func, blocksize, memsize, flags
    ):
        retval = c_int()
        args = (byref(retval), func.handle, blocksize, memsize)

        if not flags:
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(*args)
        else:
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(*args)

        return retval.value

    def get_max_potential_block_size(
        self, func, b2d_func, memsize, blocksizelimit, flags=None
    ):
        """Suggest a launch configuration with reasonable occupancy.
        :param func: kernel for which occupancy is calculated
        :param b2d_func: function that calculates how much per-block dynamic
                         shared memory 'func' uses based on the block size.
                         Can also be the address of a C function.
                         Use `0` to pass `NULL` to the underlying CUDA API.
        :param memsize: per-block dynamic shared memory usage intended, in bytes
        :param blocksizelimit: maximum block size the kernel is designed to
                               handle
        """
        args = (func, b2d_func, memsize, blocksizelimit, flags)
        return self._cuda_python_max_potential_block_size(*args)

    def _ctypes_max_potential_block_size(
        self, func, b2d_func, memsize, blocksizelimit, flags
    ):
        gridsize = c_int()
        blocksize = c_int()
        b2d_cb = cu_occupancy_b2d_size(b2d_func)
        args = [
            byref(gridsize),
            byref(blocksize),
            func.handle,
            b2d_cb,
            memsize,
            blocksizelimit,
        ]

        if not flags:
            driver.cuOccupancyMaxPotentialBlockSize(*args)
        else:
            args.append(flags)
            driver.cuOccupancyMaxPotentialBlockSizeWithFlags(*args)

        return (gridsize.value, blocksize.value)

    def _cuda_python_max_potential_block_size(
        self, func, b2d_func, memsize, blocksizelimit, flags
    ):
        b2d_cb = ctypes.CFUNCTYPE(c_size_t, c_int)(b2d_func)
        ptr = int.from_bytes(b2d_cb, byteorder="little")
        driver_b2d_cb = binding.CUoccupancyB2DSize(ptr)
        args = [func.handle, driver_b2d_cb, memsize, blocksizelimit]

        if not flags:
            return driver.cuOccupancyMaxPotentialBlockSize(*args)
        else:
            args.append(flags)
            return driver.cuOccupancyMaxPotentialBlockSizeWithFlags(*args)

    def prepare_for_use(self):
        """Initialize the context for use.
        It's safe to be called multiple times.
        """
        self.memory_manager.initialize()

    def push(self):
        """
        Pushes this context on the current CPU Thread.
        """
        driver.cuCtxPushCurrent(self.handle.value)
        self.prepare_for_use()

    def pop(self):
        """
        Pops this context off the current CPU thread. Note that this context
        must be at the top of the context stack, otherwise an error will occur.
        """
        popped = driver.pop_active_context()
        assert popped.value == self.handle.value

    def memalloc(self, bytesize):
        return self.memory_manager.memalloc(bytesize)

    def memallocmanaged(self, bytesize, attach_global=True):
        return self.memory_manager.memallocmanaged(bytesize, attach_global)

    def memhostalloc(self, bytesize, mapped=False, portable=False, wc=False):
        return self.memory_manager.memhostalloc(bytesize, mapped, portable, wc)

    def mempin(self, owner, pointer, size, mapped=False):
        if mapped and not self.device.CAN_MAP_HOST_MEMORY:
            raise CudaDriverError("%s cannot map host memory" % self.device)
        return self.memory_manager.mempin(owner, pointer, size, mapped)

    def get_ipc_handle(self, memory):
        """
        Returns an *IpcHandle* from a GPU allocation.
        """
        if not SUPPORTS_IPC:
            raise OSError("OS does not support CUDA IPC")
        return self.memory_manager.get_ipc_handle(memory)

    def open_ipc_handle(self, handle, size):
        # open the IPC handle to get the device pointer
        flags = 1  # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        dptr = driver.cuIpcOpenMemHandle(handle, flags)

        # wrap it
        return MemoryPointer(
            context=weakref.proxy(self), pointer=dptr, size=size
        )

    def enable_peer_access(self, peer_context, flags=0):
        """Enable peer access between the current context and the peer context"""
        assert flags == 0, "*flags* is reserved and MUST be zero"
        driver.cuCtxEnablePeerAccess(peer_context, flags)

    def can_access_peer(self, peer_device):
        """Returns a bool indicating whether the peer access between the
        current and peer device is possible.
        """
        peer_device = binding.CUdevice(peer_device)
        can_access_peer = driver.cuDeviceCanAccessPeer(
            self.device.id, peer_device
        )

        return bool(can_access_peer)

    def create_module_ptx(self, ptx):
        if isinstance(ptx, str):
            ptx = ptx.encode("utf8")
        image = ObjectCode.from_ptx(ptx)
        return self.create_module_image(image)

    def create_module_image(
        self, image, setup_callbacks=None, teardown_callbacks=None
    ):
        module = load_module_image(
            self, image, setup_callbacks, teardown_callbacks
        )
        key = module.handle
        self.modules[key] = module
        return weakref.proxy(module)

    def unload_module(self, module):
        key = module.handle
        del self.modules[key]

    def get_default_stream(self):
        handle = drvapi.cu_stream(int(binding.CUstream(CU_STREAM_DEFAULT)))
        return Stream(handle)

    def get_legacy_default_stream(self):
        handle = drvapi.cu_stream(
            int(binding.CUstream(binding.CU_STREAM_LEGACY))
        )
        return Stream(handle)

    def get_per_thread_default_stream(self):
        handle = drvapi.cu_stream(
            int(binding.CUstream(binding.CU_STREAM_PER_THREAD))
        )
        return Stream(handle)

    def create_stream(self):
        # The default stream creation flag, specifying that the created
        # stream synchronizes with stream 0 (this is different from the
        # default stream, which we define also as CU_STREAM_DEFAULT when
        # the NV binding is in use).
        flags = binding.CUstream_flags.CU_STREAM_DEFAULT.value
        handle = drvapi.cu_stream(int(driver.cuStreamCreate(flags)))
        return Stream(
            handle, finalizer=_stream_finalizer(self.deallocations, handle)
        )

    def create_external_stream(self, ptr):
        if not isinstance(ptr, int):
            raise TypeError("ptr for external stream must be an int")
        handle = drvapi.cu_stream(int(binding.CUstream(ptr)))
        return Stream(handle, external=True)

    def create_event(self, timing=True):
        flags = 0
        if not timing:
            flags |= enums.CU_EVENT_DISABLE_TIMING
        handle = drvapi.cu_event(int(driver.cuEventCreate(flags)))
        return Event(
            handle, finalizer=_event_finalizer(self.deallocations, handle)
        )

    def synchronize(self):
        driver.cuCtxSynchronize()

    @contextlib.contextmanager
    def defer_cleanup(self):
        with self.memory_manager.defer_cleanup():
            with self.deallocations.disable():
                yield

    def __repr__(self):
        return f"<CUDA context {self.handle} of device {self.device.id:d}>"

    def __eq__(self, other):
        if isinstance(other, Context):
            return self.handle == other.handle
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


def load_module_image(
    context, object_code, setup_callbacks=None, teardown_callbacks=None
):
    return CudaPythonModule(
        weakref.proxy(context),
        object_code,
        _module_finalizer(context, object_code),
        setup_callbacks,
        teardown_callbacks,
    )


def _alloc_finalizer(memory_manager, ptr, alloc_key, size):
    allocations = memory_manager.allocations
    deallocations = memory_manager.deallocations

    def core():
        if allocations:
            allocations.pop(alloc_key, None)
        deallocations.add_item(driver.cuMemFree, ptr, size)

    return core


def _hostalloc_finalizer(memory_manager, ptr, alloc_key, size, mapped):
    """
    Finalize page-locked host memory allocated by `context.memhostalloc`.

    This memory is managed by CUDA, and finalization entails deallocation. The
    issues noted in `_pin_finalizer` are not relevant in this case, and the
    finalization is placed in the `context.deallocations` queue along with
    finalization of device objects.

    """
    allocations = memory_manager.allocations
    deallocations = memory_manager.deallocations
    if not mapped:
        size = _SizeNotSet

    def core():
        if mapped and allocations:
            del allocations[alloc_key]
        deallocations.add_item(driver.cuMemFreeHost, ptr, size)

    return core


def _pin_finalizer(memory_manager, ptr, alloc_key, mapped):
    """
    Finalize temporary page-locking of host memory by `context.mempin`.

    This applies to memory not otherwise managed by CUDA. Page-locking can
    be requested multiple times on the same memory, and must therefore be
    lifted as soon as finalization is requested, otherwise subsequent calls to
    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
    This function therefore carries out finalization immediately, bypassing the
    `context.deallocations` queue.

    """
    allocations = memory_manager.allocations

    def core():
        if mapped and allocations:
            del allocations[alloc_key]
        driver.cuMemHostUnregister(ptr)

    return core


def _event_finalizer(deallocs, handle):
    def core():
        deallocs.add_item(driver.cuEventDestroy, handle.value)

    return core


def _stream_finalizer(deallocs, handle):
    def core():
        deallocs.add_item(driver.cuStreamDestroy, handle.value)

    return core


def _module_finalizer(context, object_code):
    dealloc = context.deallocations
    modules = context.modules
    key = object_code.handle

    def core(key=key):
        shutting_down = utils.shutting_down  # early bind

        def module_unload(handle):
            # If we are not shutting down, we must be called due to
            # Context.reset() of Context.unload_module().  Both must have
            # cleared the module reference from the context.
            assert shutting_down() or key not in modules
            driver.cuLibraryUnload(handle)

        dealloc.add_item(module_unload, key)

    return core


class _CudaIpcImpl(object):
    """Implementation of GPU IPC using CUDA driver API.
    This requires the devices to be peer accessible.
    """

    def __init__(self, parent):
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.offset = parent.offset
        # remember if the handle is already opened
        self._opened_mem = None

    def open(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self.base is not None:
            raise ValueError("opening IpcHandle from original process")

        if self._opened_mem is not None:
            raise ValueError("IpcHandle is already opened")

        mem = context.open_ipc_handle(self.handle, self.offset + self.size)
        # this object owns the opened allocation
        # note: it is required the memory be freed after the ipc handle is
        #       closed by the importing context.
        self._opened_mem = mem
        return mem.own().view(self.offset)

    def close(self):
        if self._opened_mem is None:
            raise ValueError("IpcHandle not opened")
        driver.cuIpcCloseMemHandle(self._opened_mem.handle)
        self._opened_mem = None


class _StagedIpcImpl(object):
    """Implementation of GPU IPC using custom staging logic to workaround
    CUDA IPC limitation on peer accessibility between devices.
    """

    def __init__(self, parent, source_info):
        self.parent = parent
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.source_info = source_info

    def open(self, context):
        from numba import cuda

        srcdev = Device.from_identity(self.source_info)
        srcdev_id = int(srcdev.id)

        impl = _CudaIpcImpl(parent=self.parent)
        # Open context on the source device.
        with cuda.gpus[srcdev_id]:
            source_ptr = impl.open(cuda.devices.get_context())

        # Allocate GPU buffer.
        newmem = context.memalloc(self.size)
        # Do D->D from the source peer-context
        # This performs automatic host staging
        device_to_device(newmem, source_ptr, self.size)

        # Cleanup source context
        with cuda.gpus[srcdev_id]:
            impl.close()

        return newmem

    def close(self):
        # Nothing has to be done here
        pass


class IpcHandle(object):
    """
    CUDA IPC handle. Serialization of the CUDA IPC handle object is implemented
    here.

    :param base: A reference to the original allocation to keep it alive
    :type base: MemoryPointer
    :param handle: The CUDA IPC handle, as a ctypes array of bytes.
    :param size: Size of the original allocation
    :type size: int
    :param source_info: The identity of the device on which the IPC handle was
                        opened.
    :type source_info: dict
    :param offset: The offset into the underlying allocation of the memory
                   referred to by this IPC handle.
    :type offset: int
    """

    def __init__(self, base, handle, size, source_info=None, offset=0):
        self.base = base
        self.handle = handle
        self.size = size
        self.source_info = source_info
        self._impl = None
        self.offset = offset

    def _sentry_source_info(self):
        if self.source_info is None:
            raise RuntimeError("IPC handle doesn't have source info")

    def can_access_peer(self, context):
        """Returns a bool indicating whether the active context can peer
        access the IPC handle
        """
        self._sentry_source_info()
        if self.source_info == context.device.get_device_identity():
            return True
        source_device = Device.from_identity(self.source_info)
        return context.can_access_peer(source_device.id)

    def open_staged(self, context):
        """Open the IPC by allowing staging on the host memory first."""
        self._sentry_source_info()

        if self._impl is not None:
            raise ValueError("IpcHandle is already opened")

        self._impl = _StagedIpcImpl(self, self.source_info)
        return self._impl.open(context)

    def open_direct(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self._impl is not None:
            raise ValueError("IpcHandle is already opened")

        self._impl = _CudaIpcImpl(self)
        return self._impl.open(context)

    def open(self, context):
        """Open the IPC handle and import the memory for usage in the given
        context.  Returns a raw CUDA memory pointer object.

        This is enhanced over CUDA IPC that it will work regardless of whether
        the source device is peer-accessible by the destination device.
        If the devices are peer-accessible, it uses .open_direct().
        If the devices are not peer-accessible, it uses .open_staged().
        """
        if self.source_info is None or self.can_access_peer(context):
            fn = self.open_direct
        else:
            fn = self.open_staged
        return fn(context)

    def open_array(self, context, shape, dtype, strides=None):
        """
        Similar to `.open()` but returns an device array.
        """
        from . import devicearray

        # by default, set strides to itemsize
        if strides is None:
            strides = dtype.itemsize
        dptr = self.open(context)
        # read the device pointer as an array
        return devicearray.DeviceNDArray._create_nowarn(
            shape=shape, strides=strides, dtype=dtype, gpu_data=dptr
        )

    def close(self):
        if self._impl is None:
            raise ValueError("IpcHandle not opened")
        self._impl.close()
        self._impl = None

    def __reduce__(self):
        # Preprocess the IPC handle, which is defined as a byte array.
        preprocessed_handle = self.handle.reserved
        args = (
            self.__class__,
            preprocessed_handle,
            self.size,
            self.source_info,
            self.offset,
        )
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, handle_ary, size, source_info, offset):
        handle = binding.CUipcMemHandle()
        handle.reserved = handle_ary
        return cls(
            base=None,
            handle=handle,
            size=size,
            source_info=source_info,
            offset=offset,
        )


class MemoryPointer(object):
    """A memory pointer that owns a buffer, with an optional finalizer. Memory
    pointers provide reference counting, and instances are initialized with a
    reference count of 1.

    The base ``MemoryPointer`` class does not use the
    reference count for managing the buffer lifetime. Instead, the buffer
    lifetime is tied to the memory pointer instance's lifetime:

    - When the instance is deleted, the finalizer will be called.
    - When the reference count drops to 0, no action is taken.

    Subclasses of ``MemoryPointer`` may modify these semantics, for example to
    tie the buffer lifetime to the reference count, so that the buffer is freed
    when there are no more references.

    :param context: The context in which the pointer was allocated.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the allocation in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MemoryPointer`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """

    __cuda_memory__ = True

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        if isinstance(pointer, ctypes.c_void_p):
            pointer = binding.CUdeviceptr(pointer.value)

        self.context = context
        self.device_pointer = pointer
        self.size = size
        self._cuda_memsize_ = size
        self.is_managed = finalizer is not None
        self.refct = 1
        self.handle = self.device_pointer
        self._owner = owner

        if finalizer is not None:
            self._finalizer = weakref.finalize(self, finalizer)

    @property
    def owner(self):
        return self if self._owner is None else self._owner

    def own(self):
        return OwnedPointer(weakref.proxy(self))

    def free(self):
        """
        Forces the device memory to the trash.
        """
        if self.is_managed:
            if not self._finalizer.alive:
                raise RuntimeError("Freeing dead memory")
            self._finalizer()
            assert not self._finalizer.alive

    def memset(self, byte, count=None, stream=0):
        count = self.size if count is None else count
        if stream:
            driver.cuMemsetD8Async(
                self.device_pointer, byte, count, stream.handle
            )
        else:
            driver.cuMemsetD8(self.device_pointer, byte, count)

    def view(self, start, stop=None):
        if stop is None:
            size = self.size - start
        else:
            size = stop - start

        # Handle NULL/empty memory buffer
        if not self.device_pointer_value:
            if size != 0:
                raise RuntimeError("non-empty slice into empty slice")
            view = self  # new view is just a reference to self
        # Handle normal case
        else:
            base = self.device_pointer_value + start
            if size < 0:
                raise RuntimeError("size cannot be negative")
            pointer = binding.CUdeviceptr()
            ctypes_ptr = drvapi.cu_device_ptr.from_address(pointer.getPtr())
            ctypes_ptr.value = base
            view = MemoryPointer(self.context, pointer, size, owner=self.owner)

        if isinstance(self.owner, (MemoryPointer, OwnedPointer)):
            # Owned by a numba-managed memory segment, take an owned reference
            return OwnedPointer(weakref.proxy(self.owner), view)
        else:
            # Owned by external alloc, return view with same external owner
            return view

    @property
    def device_ctypes_pointer(self):
        return drvapi.cu_device_ptr(int(self.device_pointer))

    @property
    def device_pointer_value(self):
        return int(self.device_pointer) or None


class AutoFreePointer(MemoryPointer):
    """Modifies the ownership semantic of the MemoryPointer so that the
    instance lifetime is directly tied to the number of references.

    When the reference count reaches zero, the finalizer is invoked.

    Constructor arguments are the same as for :class:`MemoryPointer`.
    """

    def __init__(self, *args, **kwargs):
        super(AutoFreePointer, self).__init__(*args, **kwargs)
        # Releease the self reference to the buffer, so that the finalizer
        # is invoked if all the derived pointers are gone.
        self.refct -= 1


class MappedMemory(AutoFreePointer):
    """A memory pointer that refers to a buffer on the host that is mapped into
    device memory.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``MappedMemory`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """

    __cuda_memory__ = True

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.owned = owner
        self.host_pointer = pointer

        devptr = driver.cuMemHostGetDevicePointer(pointer, 0)
        self._bufptr_ = self.host_pointer

        self.device_pointer = devptr
        super(MappedMemory, self).__init__(
            context, devptr, size, finalizer=finalizer
        )
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size

    def own(self):
        return MappedOwnedPointer(weakref.proxy(self))


class PinnedMemory(mviewbuf.MemAlloc):
    """A pointer to a pinned buffer on the host.

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param owner: The object owning the memory. For EMM plugin implementation,
                  this ca
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: An object owning the buffer that has been pinned. For EMM
                  plugin implementation, the default of ``None`` suffices for
                  memory allocated in ``memhostalloc`` - for ``mempin``, it
                  should be the owner passed in to the ``mempin`` method.
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.context = context
        self.owned = owner
        self.size = size
        self.host_pointer = pointer
        self.is_managed = finalizer is not None
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.host_pointer

        if finalizer is not None:
            weakref.finalize(self, finalizer)

    def own(self):
        return self


class ManagedMemory(AutoFreePointer):
    """A memory pointer that refers to a managed memory buffer (can be accessed
    on both host and device).

    :param context: The context in which the pointer was mapped.
    :type context: Context
    :param pointer: The address of the buffer.
    :type pointer: ctypes.c_void_p
    :param size: The size of the buffer in bytes.
    :type size: int
    :param owner: The owner is sometimes set by the internals of this class, or
                  used for Numba's internal memory management. It should not be
                  provided by an external user of the ``ManagedMemory`` class
                  (e.g. from within an EMM Plugin); the default of `None`
                  should always suffice.
    :type owner: NoneType
    :param finalizer: A function that is called when the buffer is to be freed.
    :type finalizer: function
    """

    __cuda_memory__ = True

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.owned = owner
        devptr = pointer
        super().__init__(context, devptr, size, finalizer=finalizer)

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.device_pointer

    def own(self):
        return ManagedOwnedPointer(weakref.proxy(self))


class OwnedPointer(object):
    def __init__(self, memptr, view=None):
        self._mem = memptr

        if view is None:
            self._view = self._mem
        else:
            assert not view.is_managed
            self._view = view

        mem = self._mem

        def deref():
            try:
                mem.refct -= 1
                assert mem.refct >= 0
                if mem.refct == 0:
                    mem.free()
            except ReferenceError:
                # ignore reference error here
                pass

        self._mem.refct += 1
        weakref.finalize(self, deref)

        # pull this attribute out for speed, because it's used often and
        # there's overhead to going through `__getattr__`
        self.device_ctypes_pointer = self._view.device_ctypes_pointer

    def __getattr__(self, fname):
        """Proxy MemoryPointer methods"""
        return getattr(self._view, fname)


class MappedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc):
    pass


class ManagedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc):
    pass


class Stream:
    def __init__(self, handle, finalizer=None, external=False):
        self.handle = handle
        self.external = external
        if finalizer is not None:
            weakref.finalize(self, finalizer)

    def __int__(self):
        # The default stream's handle.value is 0, which gives `None`
        return self.handle.value or drvapi.CU_STREAM_DEFAULT

    def __cuda_stream__(self):
        if not self.handle.value:
            return (0, drvapi.CU_STREAM_DEFAULT)
        return (0, self.handle.value)

    def __repr__(self):
        default_streams = {
            drvapi.CU_STREAM_DEFAULT: "<Default CUDA stream>",
            drvapi.CU_STREAM_LEGACY: "<Legacy default CUDA stream>",
            drvapi.CU_STREAM_PER_THREAD: "<Per-thread default CUDA stream>",
        }
        ptr = self.handle.value or drvapi.CU_STREAM_DEFAULT

        if ptr in default_streams:
            return default_streams[ptr]
        elif self.external:
            return f"<External CUDA stream {ptr:d}>"
        else:
            return f"<CUDA stream {ptr:d}>"

    def synchronize(self):
        """
        Wait for all commands in this stream to execute. This will commit any
        pending memory transfers.
        """
        handle = self.handle.value
        driver.cuStreamSynchronize(handle)

    @contextlib.contextmanager
    def auto_synchronize(self):
        """
        A context manager that waits for all commands in this stream to execute
        and commits any pending memory transfers upon exiting the context.
        """
        yield self
        self.synchronize()

    def add_callback(self, callback, arg=None):
        """
        Add a callback to a compute stream.
        The user provided function is called from a driver thread once all
        preceding stream operations are complete.

        Callback functions are called from a CUDA driver thread, not from
        the thread that invoked `add_callback`. No CUDA API functions may
        be called from within the callback function.

        The duration of a callback function should be kept short, as the
        callback will block later work in the stream and may block other
        callbacks from being executed.

        .. warning::
            There is a potential for deadlock due to a lock ordering issue
            between the GIL and the CUDA driver lock when using libraries
            that call CUDA functions without releasing the GIL. This can
            occur when the callback function, which holds the CUDA driver lock,
            attempts to acquire the GIL while another thread that holds the GIL
            is waiting for the CUDA driver lock. Consider using libraries that
            properly release the GIL around CUDA operations or restructure
            your code to avoid this situation.

        Note: The driver function underlying this method is marked for
        eventual deprecation and may be replaced in a future CUDA release.

        :param callback: Callback function with arguments (stream, status, arg).
        :param arg: Optional user data to be passed to the callback function.
        """
        data = (self, callback, arg)
        _py_incref(data)
        ptr = int.from_bytes(self._stream_callback, byteorder="little")
        stream_callback = binding.CUstreamCallback(ptr)
        # The callback needs to receive a pointer to the data PyObject
        data = id(data)
        handle = self.handle.value
        driver.cuStreamAddCallback(handle, stream_callback, data, 0)

    @staticmethod
    @cu_stream_callback_pyobj
    def _stream_callback(handle, status, data):
        try:
            stream, callback, arg = data
            callback(stream, status, arg)
        except Exception as e:
            warnings.warn(f"Exception in stream callback: {e}")
        finally:
            _py_decref(data)

    def async_done(self) -> asyncio.futures.Future:
        """
        Return an awaitable that resolves once all preceding stream operations
        are complete. The result of the awaitable is the current stream.

        .. warning::
            There is a potential for deadlock due to a lock ordering issue
            between the GIL and the CUDA driver lock when using libraries
            that call CUDA functions without releasing the GIL. This can
            occur when the callback function (internally used by this method),
            which holds the CUDA driver lock, attempts to acquire the GIL
            while another thread that holds the GIL is waiting for the CUDA driver lock.
            Consider using libraries that properly release the GIL around
            CUDA operations or restructure your code to avoid this situation.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def resolver(future, status):
            if future.done():
                return
            elif status == 0:
                future.set_result(self)
            else:
                future.set_exception(Exception(f"Stream error {status}"))

        def callback(stream, status, future):
            loop.call_soon_threadsafe(resolver, future, status)

        self.add_callback(callback, future)
        return future


def _to_core_stream(stream):
    # stream can be: int (0 for default), Stream (shim), or ExperimentalStream
    if not stream:
        return ExperimentalStream.from_handle(0)
    elif isinstance(stream, Stream):
        return ExperimentalStream.from_handle(stream.handle.value or 0)
    elif isinstance(stream, ExperimentalStream):
        return stream
    else:
        raise TypeError(
            f"Expected a Stream object, ExperimentalStream, or 0, got {type(stream).__name__}"
        )


class Event:
    def __init__(self, handle, finalizer=None):
        self.handle = handle
        if finalizer is not None:
            weakref.finalize(self, finalizer)

    def query(self):
        """
        Returns True if all work before the most recent record has completed;
        otherwise, returns False.
        """
        try:
            driver.cuEventQuery(self.handle)
        except CudaAPIError as e:
            if e.code == enums.CUDA_ERROR_NOT_READY:
                return False
            else:
                raise
        else:
            return True

    def record(self, stream=0):
        """
        Set the record point of the event to the current point in the given
        stream.

        The event will be considered to have occurred when all work that was
        queued in the stream at the time of the call to ``record()`` has been
        completed.
        """
        hstream = _stream_handle(stream)
        handle = self.handle.value
        driver.cuEventRecord(handle, hstream)

    def synchronize(self):
        """
        Synchronize the host thread for the completion of the event.
        """
        handle = self.handle.value
        driver.cuEventSynchronize(handle)

    def wait(self, stream=0):
        """
        All future works submitted to stream will wait util the event completes.
        """
        hstream = _stream_handle(stream)
        handle = self.handle.value
        flags = 0
        driver.cuStreamWaitEvent(hstream, handle, flags)

    def elapsed_time(self, evtend):
        return event_elapsed_time(self, evtend)


def event_elapsed_time(evtstart, evtend):
    """
    Compute the elapsed time between two events in milliseconds.
    """
    return driver.cuEventElapsedTime(evtstart.handle.value, evtend.handle.value)


class CudaPythonModule:
    def __init__(
        self,
        context,
        object_code,
        finalizer=None,
        setup_callbacks=None,
        teardown_callbacks=None,
    ):
        self.context = context
        self.object_code = object_code
        self.handle = object_code.handle
        if finalizer is not None:
            self._finalizer = weakref.finalize(self, finalizer)

        self.initialized = False
        self.setup_functions = setup_callbacks
        self.teardown_functions = teardown_callbacks

        self._set_finalizers()

    def unload(self):
        """Unload this module from the context"""
        self.context.unload_module(self)

    def setup(self):
        """Call the setup functions for the module"""
        if self.initialized:
            raise RuntimeError("The module has already been initialized.")

        if self.setup_functions is None:
            return

        for f in self.setup_functions:
            f(self.object_code)

        self.initialized = True

    def _set_finalizers(self):
        """Create finalizers that tear down the module."""
        if self.teardown_functions is None:
            return

        def _teardown(teardowns, object_code):
            for f in teardowns:
                f(object_code)

        weakref.finalize(
            self,
            _teardown,
            self.teardown_functions,
            self.object_code,
        )

    def get_function(self, name):
        """Returns a Function object encapsulating the named function"""
        kernel = self.object_code.get_kernel(name)
        return Function(weakref.proxy(self), kernel, name)

    def get_global_symbol(self, name):
        """Return a MemoryPointer referring to the named symbol"""
        ptr, size = driver.cuLibraryGetGlobal(self.handle, name.encode("utf8"))
        return MemoryPointer(self.context, ptr, size), size


FuncAttr = namedtuple(
    "FuncAttr", ["regs", "shared", "local", "const", "maxthreads"]
)


class CudaPythonFunction:
    griddim = 1, 1, 1
    blockdim = 1, 1, 1
    stream = 0
    sharedmem = 0

    __slots__ = "module", "kernel", "handle", "name", "attrs"

    def __init__(self, module, kernel, name):
        self.module = module
        self.kernel = kernel
        self.handle = kernel._handle
        self.name = name
        attrs = self.kernel.attributes
        self.attrs = FuncAttr(
            regs=attrs.num_regs(),
            const=attrs.const_size_bytes(),
            local=attrs.local_size_bytes(),
            shared=attrs.shared_size_bytes(),
            maxthreads=attrs.max_threads_per_block(),
        )

    def __repr__(self):
        return "<CUDA function %s>" % self.name

    @property
    def device(self):
        return self.module.context.device

    def cache_config(
        self, prefer_equal=False, prefer_cache=False, prefer_shared=False
    ):
        prefer_equal = prefer_equal or (prefer_cache and prefer_shared)
        attr = binding.CUfunc_cache
        if prefer_equal:
            flag = attr.CU_FUNC_CACHE_PREFER_EQUAL
        elif prefer_cache:
            flag = attr.CU_FUNC_CACHE_PREFER_L1
        elif prefer_shared:
            flag = attr.CU_FUNC_CACHE_PREFER_SHARED
        else:
            flag = attr.CU_FUNC_CACHE_PREFER_NONE
        driver.cuKernelSetCacheConfig(self.handle, flag, self.device.id)

    def set_shared_memory_carveout(self, carveout):
        carveout = int(carveout)

        if not (-1 <= carveout <= 100):
            raise ValueError("Carveout must be between -1 and 100")

        attr = binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        driver.cuKernelSetAttribute(attr, carveout, self.handle, self.device.id)


# Alias for backward compatibility
Function = CudaPythonFunction


class _Linker:
    def __init__(
        self,
        max_registers=None,
        lineinfo=False,
        cc=None,
        lto=None,
        additional_flags=None,
    ):
        arch = f"sm_{cc[0]}{cc[1]}"
        self.max_registers = max_registers if max_registers else None
        self.lineinfo = lineinfo
        self.cc = cc
        self.arch = arch
        if lto is False:
            # WAR for apparent nvjitlink issue
            lto = None
        self.lto = lto
        self.additional_flags = additional_flags

        self.options = LinkerOptions(
            max_register_count=self.max_registers,
            lineinfo=lineinfo,
            arch=arch,
            link_time_optimization=lto,
        )
        self._complete = False
        self._object_codes = []
        self.linker = None  # need at least one program

    def add_cu_file(self, path):
        cu = cached_file_read(path, how="rb")
        self.add_cu(cu, os.path.basename(path))

    def add_file_guess_ext(self, path_or_code, ignore_nonlto=False):
        """
        Add a file or LinkableCode object to the link. If a file is
        passed, the type will be inferred from the extension. A LinkableCode
        object represents a file already in memory.

        When `ignore_nonlto` is set to true, do not add code that will not
        be LTO-ed in the linking process. This is useful in inspecting the
        LTO-ed portion of the PTX when linker is added with objects that can be
        both LTO-ed and not LTO-ed.
        """
        if isinstance(path_or_code, str):
            ext = pathlib.Path(path_or_code).suffix
            if ext == "":
                raise RuntimeError(
                    "Don't know how to link file with no extension"
                )
            elif ext == ".cu":
                self.add_cu_file(path_or_code)
            else:
                kind = FILE_EXTENSION_MAP.get(ext.lstrip("."), None)
                if kind is None:
                    raise RuntimeError(
                        f"Don't know how to link file with extension {ext}"
                    )

                if ignore_nonlto:
                    warn_and_return = False
                    if kind in (
                        FILE_EXTENSION_MAP["fatbin"],
                        FILE_EXTENSION_MAP["o"],
                    ):
                        entry_types = inspect_obj_content(path_or_code)
                        if "nvvm" not in entry_types:
                            warn_and_return = True
                    elif kind != FILE_EXTENSION_MAP["ltoir"]:
                        warn_and_return = True

                    if warn_and_return:
                        warnings.warn(
                            f"Not adding {path_or_code} as it is not "
                            "optimizable at link time, and `ignore_nonlto == "
                            "True`."
                        )
                        return

                self.add_file(path_or_code, kind)
            return
        else:
            # Otherwise, we should have been given a LinkableCode object
            if not isinstance(path_or_code, LinkableCode):
                raise TypeError(
                    "Expected path to file or a LinkableCode object"
                )

            if path_or_code.kind == "cu":
                self.add_cu(path_or_code.data, path_or_code.name)
            else:
                if ignore_nonlto:
                    warn_and_return = False
                    if isinstance(path_or_code, (Fatbin, Object)):
                        with tempfile.NamedTemporaryFile("w") as fp:
                            fp.write(path_or_code.data)
                            entry_types = inspect_obj_content(fp.name)
                        if "nvvm" not in entry_types:
                            warn_and_return = True
                    elif not isinstance(path_or_code, LTOIR):
                        warn_and_return = True

                    if warn_and_return:
                        warnings.warn(
                            f"Not adding {path_or_code.name} as it is not "
                            "optimizable at link time, and `ignore_nonlto == "
                            "True`."
                        )
                        return

                self.add_data(
                    path_or_code.data, path_or_code.kind, path_or_code.name
                )

    @property
    def info_log(self):
        """Return the info log from the linker invocation"""
        if not self.linker:
            raise ValueError("Not Initialized")
        if self._complete:
            return self._info_log
        raise RuntimeError("Link not yet complete.")

    @property
    def error_log(self):
        """Return the error log from the linker invocation"""
        if not self.linker:
            raise ValueError("Not Initialized")
        if self._complete:
            return self._error_log
        raise RuntimeError("Link not yet complete.")

    def add_ptx(self, ptx, name="<cudapy-ptx>"):
        """Add PTX source in a string to the link"""
        obj = ObjectCode.from_ptx(ptx, name=name)
        self._object_codes.append(obj)

    def add_cu(self, cu, name="<cudapy-cu>"):
        """Add CUDA source in a string to the link. The name of the source
        file should be specified in `name`."""
        obj, log = nvrtc.compile(cu, name, self.cc, ltoir=self.lto)

        if not self.lto and config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % name).center(80, "-"))
            print(obj.code)

        self._object_codes.append(obj)

    def add_cubin(self, cubin, name="<cudapy-cubin>"):
        obj = ObjectCode.from_cubin(cubin, name=name)
        self._object_codes.append(obj)

    def add_ltoir(self, ltoir, name="<cudapy-ltoir>"):
        obj = ObjectCode.from_ltoir(ltoir, name=name)
        self._object_codes.append(obj)

    def add_fatbin(self, fatbin, name="<cudapy-fatbin>"):
        obj = ObjectCode.from_fatbin(fatbin, name=name)
        self._object_codes.append(obj)

    def add_object(self, obj, name="<cudapy-object>"):
        obj = ObjectCode.from_object(obj, name=name)
        self._object_codes.append(obj)

    def add_library(self, lib, name="<cudapy-lib>"):
        obj = ObjectCode.from_library(lib, name=name)
        self._object_codes.append(obj)

    def add_file(self, path, kind):
        """Add code from a file to the link"""
        try:
            data = cached_file_read(path, how="rb")
        except FileNotFoundError:
            raise LinkerError(f"{path} not found")
        name = pathlib.Path(path).name
        self.add_data(data, kind, name)

    def add_data(self, data, kind, name):
        """Add in-memory data to the link"""
        if kind == FILE_EXTENSION_MAP["ptx"]:
            fn = self.add_ptx
        elif kind == FILE_EXTENSION_MAP["cubin"]:
            fn = self.add_cubin
        elif kind == "cu":
            fn = self.add_cu
        elif (
            kind == FILE_EXTENSION_MAP["lib"] or kind == FILE_EXTENSION_MAP["a"]
        ):
            fn = self.add_library
        elif kind == FILE_EXTENSION_MAP["fatbin"]:
            fn = self.add_fatbin
        elif kind == FILE_EXTENSION_MAP["o"]:
            fn = self.add_object
        elif kind == FILE_EXTENSION_MAP["ltoir"]:
            fn = self.add_ltoir
        else:
            raise LinkerError(f"Don't know how to link {kind}")

        fn(data, name)

    def get_linked_ptx(self):
        options = LinkerOptions(
            max_register_count=self.max_registers,
            lineinfo=self.lineinfo,
            arch=self.arch,
            link_time_optimization=True,
            ptx=True,
        )

        self.linker = Linker(*self._object_codes, options=options)

        result = self.linker.link("ptx")
        self.close()
        self._complete = True
        return result.code

    def close(self):
        self._info_log = self.linker.get_info_log()
        self._error_log = self.linker.get_error_log()
        self.linker.close()

    def complete(self):
        """Complete the link. Returns (cubin, size)

        cubin is a pointer to a internal buffer of cubin owned by the linker;
        thus, it should be loaded before the linker is destroyed.
        """
        self.linker = Linker(*self._object_codes, options=self.options)
        result = self.linker.link("cubin")
        self.close()
        self._complete = True
        return result


# -----------------------------------------------------------------------------


def get_devptr_for_active_ctx(ptr):
    """Query the device pointer usable in the current context from an arbitrary
    pointer.
    """
    if ptr != 0:
        ptr_attrs = binding.CUpointer_attribute
        attr = ptr_attrs.CU_POINTER_ATTRIBUTE_DEVICE_POINTER
        ptrobj = binding.CUdeviceptr(ptr)
        return driver.cuPointerGetAttribute(attr, ptrobj)
    else:
        return binding.CUdeviceptr()


def device_extents(devmem):
    """Find the extents (half open begin and end pointer) of the underlying
    device memory allocation.

    NOTE: it always returns the extents of the allocation but the extents
    of the device memory view that can be a subsection of the entire allocation.
    """
    devptr = device_ctypes_pointer(devmem)
    s, n = driver.cuMemGetAddressRange(devptr.value)
    return int(s), int(binding.CUdeviceptr(int(s) + n))


def device_memory_size(devmem):
    """Check the memory size of the device memory.
    The result is cached in the device memory object.
    It may query the driver for the memory size of the device memory allocation.
    """
    sz = getattr(devmem, "_cuda_memsize_", None)
    if sz is None:
        s, e = device_extents(devmem)
        sz = e - s
        devmem._cuda_memsize_ = sz
    assert sz >= 0, "{} length array".format(sz)
    return sz


def _is_datetime_dtype(obj):
    """Returns True if the obj.dtype is datetime64 or timedelta64"""
    dtype = getattr(obj, "dtype", None)
    return dtype is not None and dtype.char in "Mm"


def _workaround_for_datetime(obj):
    """Workaround for numpy#4983: buffer protocol doesn't support
    datetime64 or timedelta64.
    """
    if _is_datetime_dtype(obj):
        obj = obj.view(np.int64)
    return obj


def host_pointer(obj, readonly=False):
    """Get host pointer from an obj.

    If `readonly` is False, the buffer must be writable.

    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    if isinstance(obj, int):
        return obj

    forcewritable = False
    if not readonly:
        forcewritable = isinstance(obj, np.void) or _is_datetime_dtype(obj)

    obj = _workaround_for_datetime(obj)
    return mviewbuf.memoryview_get_buffer(obj, forcewritable, readonly)


def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    obj = _workaround_for_datetime(obj)
    return mviewbuf.memoryview_get_extents(obj)


@functools.cache
def memory_size_from_info(shape, strides, itemsize):
    """Get the byte size of a contiguous memory buffer given the shape, strides
    and itemsize.
    """
    assert len(shape) == len(strides), "# dim mismatch"
    ndim = len(shape)
    s, e = mviewbuf.memoryview_get_extents_info(shape, strides, ndim, itemsize)
    return e - s


def host_memory_size(obj):
    "Get the size of the memory"
    s, e = host_memory_extents(obj)
    assert e >= s, "memory extend of negative size"
    return e - s


def device_pointer(obj):
    "Get the device pointer as an integer"
    return device_ctypes_pointer(obj).value


def device_ctypes_pointer(obj):
    "Get the ctypes object for the device pointer"
    if obj is None:
        return c_void_p(0)
    require_device_memory(obj)
    return obj.device_ctypes_pointer


def is_device_memory(obj):
    """All CUDA memory object is recognized as an instance with the attribute
    "__cuda_memory__" defined and its value evaluated to True.

    All CUDA memory object should also define an attribute named
    "device_pointer" which value is an int object carrying the pointer
    value of the device memory address.  This is not tested in this method.
    """
    try:
        # This is cheaper than getattr in the non-exceptional case
        return obj.__cuda_memory__
    except AttributeError:
        return False


def require_device_memory(obj):
    """A sentry for methods that accept CUDA memory object."""
    if not is_device_memory(obj):
        raise Exception("Not a CUDA memory object.")


def device_memory_depends(devmem, *objs):
    """Add dependencies to the device memory.

    Mainly used for creating structures that points to other device memory,
    so that the referees are not GC and released.
    """
    depset = getattr(devmem, "_depends_", [])
    depset.extend(objs)


def host_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    fn = driver.cuMemcpyHtoD
    args = (device_pointer(dst), host_pointer(src, readonly=True), size)

    if stream:
        fn = driver.cuMemcpyHtoDAsync
        args += (_stream_handle(stream),)

    fn(*args)


def device_to_host(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    fn = driver.cuMemcpyDtoH
    args = (host_pointer(dst), device_pointer(src), size)

    if stream:
        fn = driver.cuMemcpyDtoHAsync
        args += (_stream_handle(stream),)

    fn(*args)


def device_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the device buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    fn = driver.cuMemcpyDtoD
    args = (device_pointer(dst), device_pointer(src), size)

    if stream:
        fn = driver.cuMemcpyDtoDAsync
        args += (_stream_handle(stream),)

    fn(*args)


def device_memset(dst, val, size, stream=0):
    """
    Memset on the device.
    If stream is 0, the call is synchronous.
    If stream is a Stream object, asynchronous mode is used.

    dst: device memory
    val: byte value to be written
    size: number of bytes to be written
    stream: 0 (synchronous) or a CUDA stream
    """
    fn = driver.cuMemsetD8
    args = (device_pointer(dst), val, size)

    if stream:
        fn = driver.cuMemsetD8Async
        args += (_stream_handle(stream),)

    try:
        fn(*args)
    except CudaAPIError as e:
        invalid = binding.CUresult.CUDA_ERROR_INVALID_VALUE
        if (
            e.code == invalid
            and getattr(dst, "__cuda_memory__", False)
            and getattr(dst, "is_managed", False)
        ):
            buf = (c_uint8 * size).from_address(host_pointer(dst))
            byte = val & 0xFF
            buf[:] = [byte] * size
            return
        raise


def profile_start():
    """
    Enable profile collection in the current context.
    """
    driver.cuProfilerStart()


def profile_stop():
    """
    Disable profile collection in the current context.
    """
    driver.cuProfilerStop()


@contextlib.contextmanager
def profiling():
    """
    Context manager that enables profiling on entry and disables profiling on
    exit.
    """
    profile_start()
    yield
    profile_stop()


def get_version():
    """
    Return the driver version as a tuple of (major, minor)
    """
    return driver.get_version()


def inspect_obj_content(objpath: str):
    """
    Given path to a fatbin or object, use `cuobjdump` to examine its content
    Return the set of entries in the object.
    """
    code_types: set[str] = set()

    try:
        out = subprocess.run(
            ["cuobjdump", objpath], check=True, capture_output=True
        )
    except FileNotFoundError as e:
        msg = (
            "cuobjdump has not been found. You may need "
            "to install the CUDA toolkit and ensure that "
            "it is available on your PATH.\n"
        )
        raise RuntimeError(msg) from e

    objtable = out.stdout.decode("utf-8")
    entry_pattern = r"Fatbin (.*) code"
    for line in objtable.split("\n"):
        if match := re.match(entry_pattern, line):
            code_types.add(match.group(1))

    return code_types


def _stream_handle(stream):
    """
    Obtain the appropriate handle for various types of
    acceptable stream objects. Acceptable types are
    int (0 for default stream), Stream, ExperimentalStream
    """

    if stream == 0:
        return stream
    allowed = (Stream, ExperimentalStream)
    if not isinstance(stream, allowed):
        raise TypeError(
            "Expected a Stream object or 0, got %s" % type(stream).__name__
        )
    elif hasattr(stream, "__cuda_stream__"):
        ver, ptr = stream.__cuda_stream__()
        assert ver == 0
        if isinstance(ptr, binding.CUstream):
            return get_cuda_native_handle(ptr)
        else:
            return ptr
    else:
        raise TypeError("Invalid Stream")
