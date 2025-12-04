# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
API that are reported to numba.cuda
"""

import contextlib

import numpy as np
import warnings
from .cudadrv import devicearray, devices, driver
from numba.cuda.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.cuda.cudadrv.devicearray import DeprecatedDeviceArrayApiWarning, DeviceNDArray

# NDarray device helper

require_context = devices.require_context
current_context = devices.get_context
gpus = devices.gpus

@require_context
def external_stream(ptr):
    """Create a Numba stream object for a stream allocated outside Numba.

    :param ptr: Pointer to the external stream to wrap in a Numba Stream
    :type ptr: int
    """
    return current_context().create_external_stream(ptr)

def _from_cuda_array_interface(desc, owner=None, sync=True):
    """Create a _DeviceNDArray from a cuda-array-interface description.
    The ``owner`` is the owner of the underlying memory.
    The resulting _DeviceNDArray will acquire a reference from it.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
    version = desc.get("version")
    # Mask introduced in version 1
    if 1 <= version:
        mask = desc.get("mask")
        # Would ideally be better to detect if the mask is all valid
        if mask is not None:
            raise NotImplementedError("Masked arrays are not supported")

    shape = desc["shape"]
    strides = desc.get("strides")

    shape, strides, dtype = prepare_shape_strides_dtype(
        shape, strides, desc["typestr"], order="C"
    )
    size = driver.memory_size_from_info(shape, strides, dtype.itemsize)

    cudevptr_class = driver.binding.CUdeviceptr
    devptr = cudevptr_class(desc["data"][0])
    data = driver.MemoryPointer(
        current_context(), devptr, size=size, owner=owner
    )
    stream_ptr = desc.get("stream", None)
    if stream_ptr is not None:
        stream = external_stream(stream_ptr)
        if sync and config.CUDA_ARRAY_INTERFACE_SYNC:
            stream.synchronize()
    else:
        stream = 0  # No "Numba default stream", not the CUDA default stream
    da = devicearray.DeviceNDArray._create_nowarn(
        shape=shape, strides=strides, dtype=dtype, gpu_data=data, stream=stream
    )
    return da


def _as_cuda_array(obj, sync=True):
    """Create a _DeviceNDArray from any object that implements
    the :ref:`cuda array interface <cuda-array-interface>`.

    A view of the underlying GPU buffer is created.  No copying of the data
    is done.  The resulting _DeviceNDArray will acquire a reference from `obj`.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
    if (
        interface := getattr(obj, "__cuda_array_interface__", None)
    ) is not None:
        return _from_cuda_array_interface(interface, owner=obj, sync=sync)
    raise TypeError("*obj* doesn't implement the cuda array interface.")


def _is_cuda_array(obj):
    """Test if the object has defined the `__cuda_array_interface__` attribute.

    Does not verify the validity of the interface.
    """
    return hasattr(obj, "__cuda_array_interface__")


@require_context
def _to_device(obj, stream=0, copy=True, to=None):
    """to_device(obj, stream=0, copy=True, to=None)

    Allocate and transfer a numpy ndarray or structured scalar to the device.

    To copy host->device a numpy array::

        ary = np.arange(10)
        d_ary = cuda.to_device(ary)

    To enqueue the transfer to a stream::

        stream = cuda.stream()
        d_ary = cuda.to_device(ary, stream=stream)

    The resulting ``d_ary`` is a ``DeviceNDArray``.

    To copy device->host::

        hary = d_ary.copy_to_host()

    To copy device->host to an existing array::

        ary = np.empty(shape=d_ary.shape, dtype=d_ary.dtype)
        d_ary.copy_to_host(ary)

    To enqueue the transfer to a stream::

        hary = d_ary.copy_to_host(stream=stream)
    """
    if to is None:
        to, new = devicearray.auto_device(
            obj, stream=stream, copy=copy, user_explicit=True
        )
        return to
    if copy:
        to.copy_to_device(obj, stream=stream)
    return to


@require_context
def _device_array(shape, dtype=np.float64, strides=None, order="C", stream=0):
    """device_array(shape, dtype=np.float64, strides=None, order='C', stream=0)

    Allocate an empty device ndarray. Similar to :meth:`numpy.empty`.
    """
    shape, strides, dtype = prepare_shape_strides_dtype(
        shape, strides, dtype, order
    )
    return DeviceNDArray._create_nowarn(
        shape=shape, strides=strides, dtype=dtype, stream=stream
    )


@require_context
def _managed_array(
    shape,
    dtype=np.float64,
    strides=None,
    order="C",
    stream=0,
    attach_global=True,
):
    """managed_array(shape, dtype=np.float64, strides=None, order='C', stream=0,
                     attach_global=True)

    Allocate a np.ndarray with a buffer that is managed.
    Similar to np.empty().

    Managed memory is supported on Linux / x86 and PowerPC, and is considered
    experimental on Windows and Linux / AArch64.

    :param attach_global: A flag indicating whether to attach globally. Global
                          attachment implies that the memory is accessible from
                          any stream on any device. If ``False``, attachment is
                          *host*, and memory is only accessible by devices
                          with Compute Capability 6.0 and later.
    """
    shape, strides, dtype = prepare_shape_strides_dtype(
        shape, strides, dtype, order
    )
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = current_context().memallocmanaged(
        bytesize, attach_global=attach_global
    )
    npary = np.ndarray(
        shape=shape, strides=strides, dtype=dtype, order=order, buffer=buffer
    )
    managedview = np.ndarray.view(npary, type=devicearray.ManagedNDArray)
    managedview.device_setup(buffer, stream=stream)
    return managedview


@require_context
def _pinned_array(shape, dtype=np.float64, strides=None, order="C"):
    """pinned_array(shape, dtype=np.float64, strides=None, order='C')

    Allocate an :class:`ndarray <numpy.ndarray>` with a buffer that is pinned
    (pagelocked).  Similar to :func:`np.empty() <numpy.empty>`.
    """
    warnings.warn(
        "pinned_array is deprecated. Please prefer cupy for moving numpy arrays to the device.",
        DeprecatedDeviceArrayApiWarning,
    )
    shape, strides, dtype = prepare_shape_strides_dtype(
        shape, strides, dtype, order
    )
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = current_context().memhostalloc(bytesize)
    return np.ndarray(
        shape=shape, strides=strides, dtype=dtype, order=order, buffer=buffer
    )


@require_context
def _mapped_array(
    shape,
    dtype=np.float64,
    strides=None,
    order="C",
    stream=0,
    portable=False,
    wc=False,
):
    """mapped_array(shape, dtype=np.float64, strides=None, order='C', stream=0,
                    portable=False, wc=False)

    Allocate a mapped ndarray with a buffer that is pinned and mapped on
    to the device. Similar to np.empty()

    :param portable: a boolean flag to allow the allocated device memory to be
              usable in multiple devices.
    :param wc: a boolean flag to enable writecombined allocation which is faster
        to write by the host and to read by the device, but slower to
        write by the host and slower to write by the device.
    """
    warnings.warn(
        "mapped_array is deprecated. Please prefer cupy for moving numpy arrays to the device.",
        DeprecatedDeviceArrayApiWarning,
    )
    shape, strides, dtype = prepare_shape_strides_dtype(
        shape, strides, dtype, order
    )
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = current_context().memhostalloc(bytesize, mapped=True)
    npary = np.ndarray(
        shape=shape, strides=strides, dtype=dtype, order=order, buffer=buffer
    )
    mappedview = np.ndarray.view(npary, type=devicearray.MappedNDArray)
    mappedview.device_setup(buffer, stream=stream)
    return mappedview


@contextlib.contextmanager
@require_context
def _open_ipc_array(handle, shape, dtype, strides=None, offset=0):
    """
    A context manager that opens a IPC *handle* (*CUipcMemHandle*) that is
    represented as a sequence of bytes (e.g. *bytes*, tuple of int)
    and represent it as an array of the given *shape*, *strides* and *dtype*.
    The *strides* can be omitted.  In that case, it is assumed to be a 1D
    C contiguous array.

    Yields a device array.

    The IPC handle is closed automatically when context manager exits.
    """
    dtype = np.dtype(dtype)
    # compute size
    size = np.prod(shape) * dtype.itemsize
    # manually recreate the IPC mem handle
    driver_handle = driver.binding.CUipcMemHandle()
    driver_handle.reserved = handle
    # use *IpcHandle* to open the IPC memory
    ipchandle = driver.IpcHandle(None, driver_handle, size, offset=offset)
    yield ipchandle.open_array(
        current_context(), shape=shape, strides=strides, dtype=dtype
    )
    ipchandle.close()


def _contiguous_strides_like_array(ary):
    """
    Given an array, compute strides for a new contiguous array of the same
    shape.
    """
    # Don't recompute strides if the default strides will be sufficient to
    # create a contiguous array.
    if ary.flags["C_CONTIGUOUS"] or ary.flags["F_CONTIGUOUS"] or ary.ndim <= 1:
        return None

    # Otherwise, we need to compute new strides using an algorithm adapted from
    # NumPy v1.17.4's PyArray_NewLikeArrayWithShape in
    # core/src/multiarray/ctors.c. We permute the strides in ascending order
    # then compute the stride for the dimensions with the same permutation.

    # Stride permutation. E.g. a stride array (4, -2, 12) becomes
    # [(1, -2), (0, 4), (2, 12)]
    strideperm = [x for x in enumerate(ary.strides)]
    strideperm.sort(key=lambda x: x[1])

    # Compute new strides using permutation
    strides = [0] * len(ary.strides)
    stride = ary.dtype.itemsize
    for i_perm, _ in strideperm:
        strides[i_perm] = stride
        stride *= ary.shape[i_perm]
    return tuple(strides)


def _order_like_array(ary):
    if ary.flags["F_CONTIGUOUS"] and not ary.flags["C_CONTIGUOUS"]:
        return "F"
    else:
        return "C"


def _device_array_like(ary, stream=0):
    """
    Call :func:`device_array() <numba.cuda.device_array>` with information from
    the array.
    """
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return device_array(
        shape=ary.shape,
        dtype=ary.dtype,
        strides=strides,
        order=order,
        stream=stream,
    )


def _mapped_array_like(ary, stream=0, portable=False, wc=False):
    """
    Call :func:`mapped_array() <numba.cuda.mapped_array>` with the information
    from the array.
    """
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return mapped_array(
        shape=ary.shape,
        dtype=ary.dtype,
        strides=strides,
        order=order,
        stream=stream,
        portable=portable,
        wc=wc,
    )


def _pinned_array_like(ary):
    """
    Call :func:`pinned_array() <numba.cuda.pinned_array>` with the information
    from the array.
    """
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return pinned_array(
        shape=ary.shape, dtype=ary.dtype, strides=strides, order=order
    )
