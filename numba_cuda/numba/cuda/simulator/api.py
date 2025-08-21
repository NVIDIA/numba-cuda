# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Contains CUDA API functions
"""

# Imports here bring together parts of the API from other modules, so some of
# them appear unused.
from contextlib import contextmanager

from .cudadrv.devices import require_context, reset, gpus  # noqa: F401
from .cudadrv.linkable_code import (
    PTXSource,  # noqa: F401
    CUSource,  # noqa: F401
    Cubin,  # noqa: F401
    Fatbin,  # noqa: F401
    Archive,  # noqa: F401
    Object,  # noqa: F401
    LTOIR,  # noqa: F401
)  # noqa: F401
from .kernel import FakeCUDAKernel
from numba.core import config
from numba.cuda.core.sigutils import is_signature
from warnings import warn
from ..args import In, Out, InOut  # noqa: F401


def select_device(dev=0):
    assert dev == 0, "Only a single device supported by the simulator"


def is_float16_supported():
    return True


def is_bfloat16_supported():
    return False


class stream(object):
    """
    The stream API is supported in the simulator - however, all execution
    occurs synchronously, so synchronization requires no operation.
    """

    @contextmanager
    def auto_synchronize(self):
        yield

    def synchronize(self):
        pass


# Default stream APIs. Since execution from the perspective of the host is
# synchronous in the simulator, these can be the same as the stream class.
default_stream = stream
legacy_default_stream = stream
per_thread_default_stream = stream


# There is no way to use external streams with the simulator. Since the
# implementation is not really using streams, we can't meaningfully interact
# with external ones.
def external_stream(ptr):
    raise RuntimeError("External streams are unsupported in the simulator")


def synchronize():
    pass


def close():
    gpus.closed = True


def declare_device(*args, **kwargs):
    pass


def detect():
    print("Found 1 CUDA devices")
    print("id %d    %20s %40s" % (0, "SIMULATOR", "[SUPPORTED]"))
    print("%40s: 5.0" % "compute capability")


def list_devices():
    return gpus


def get_current_device():
    return gpus[0].device


# Events


class Event(object):
    """
    The simulator supports the event API, but they do not record timing info,
    and all simulation is synchronous. Execution time is not recorded.
    """

    def record(self, stream=0):
        pass

    def wait(self, stream=0):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, event):
        warn("Simulator timings are bogus")
        return 0.0


event = Event


def jit(
    func_or_sig=None,
    device=False,
    debug=None,
    argtypes=None,
    inline=False,
    restype=None,
    fastmath=False,
    link=None,
    boundscheck=None,
    opt=None,
    cache=None,
):
    # Here for API compatibility
    if boundscheck:
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if link is not None:
        raise NotImplementedError("Cannot link PTX in the simulator")

    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug

    # Check for first argument specifying types - in that case the
    # decorator is not being passed a function
    if (
        func_or_sig is None
        or is_signature(func_or_sig)
        or isinstance(func_or_sig, list)
    ):

        def jitwrapper(fn):
            return FakeCUDAKernel(
                fn, device=device, fastmath=fastmath, debug=debug
            )

        return jitwrapper
    return FakeCUDAKernel(func_or_sig, device=device, debug=debug)


@contextmanager
def defer_cleanup():
    # No effect for simulator
    yield
