# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Hints to wrap Kernel arguments to indicate how to manage host-device
memory transfers before & after the kernel call.
"""

import functools


from numba.cuda.typing.typeof import typeof, Purpose


class ArgHint:
    def __init__(self, value):
        self.value = value

    def to_device(self, retr, stream=0):
        """
        :param stream: a stream to use when copying data
        :param retr:
            a list of clean-up work to do after the kernel's been run.
            Append 0-arg lambdas to it!
        :return: a value (usually an `DeviceNDArray`) to be passed to
            the kernel
        """

    @functools.cached_property
    def _numba_type_(self):
        return typeof(self.value, Purpose.argument)


class In(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import _to_strided_memory_view

        devary, _ = _to_strided_memory_view(self.value, stream=stream)
        # A dummy writeback functor to keep devary alive until the kernel
        # is called.
        retr.append(lambda: devary)
        return devary


class Out(ArgHint):
    copy_input = False

    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import _to_strided_memory_view
        from .cudadrv.devicearray import _make_strided_memory_view
        from .cudadrv.driver import driver
        from .cudadrv import devices

        devary, conv = _to_strided_memory_view(
            value := self.value, copy=self.__class__.copy_input, stream=stream
        )
        if conv:
            stream_ptr = getattr(stream, "handle", stream)

            def copy_to_host(devary=devary, value=value, stream_ptr=stream_ptr):
                hostary = _make_strided_memory_view(
                    value, stream_ptr=stream_ptr
                )
                nbytes = devary.size * devary.dtype.itemsize
                hostptr = hostary.ptr
                devptr = devary.ptr
                if int(stream_ptr):
                    driver.cuMemcpyDtoHAsync(
                        hostptr, devptr, nbytes, stream_ptr
                    )
                else:
                    driver.cuMemcpyDtoH(hostptr, devptr, nbytes)
                    ctx = devices.get_context()
                    stream = ctx.get_default_stream()
                    stream.synchronize()
                return hostary

            retr.append(copy_to_host)
        return devary


class InOut(Out):
    copy_input = True


def wrap_arg(value, default=InOut):
    return value if isinstance(value, ArgHint) else default(value)


__all__ = [
    "In",
    "Out",
    "InOut",
    "ArgHint",
    "wrap_arg",
]
