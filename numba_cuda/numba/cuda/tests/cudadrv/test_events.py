# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda, int32
from numba.cuda.testing import unittest, CUDATestCase
from cuda.core import Device
from numba.cuda.testing import skip_on_cudasim


class TestCudaEvent(CUDATestCase):
    def test_event_elapsed(self):
        N = 32
        dary = cuda.device_array(N, dtype=np.double)
        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record()
        cuda.to_device(np.arange(N, dtype=np.double), to=dary)
        evtend.record()
        evtend.wait()
        evtend.synchronize()
        # Exercise the code path
        evtstart.elapsed_time(evtend)

    def test_event_elapsed_stream(self):
        stream = cuda.stream()
        self.event_elapsed_inner(stream)

    @skip_on_cudasim("Testing cuda.core events requires driver")
    def test_event_elapsed_cuda_core_stream(self):
        dev = Device()
        dev.set_current()
        stream = dev.create_stream()
        self.event_elapsed_inner(stream)

    def event_elapsed_inner(self, stream):
        N = 32
        dary = cuda.device_array(N, dtype=np.double)
        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record(stream=stream)
        cuda.to_device(np.arange(N, dtype=np.double), to=dary, stream=stream)
        evtend.record(stream=stream)
        evtend.wait(stream=stream)
        evtend.synchronize()
        # Exercise the code path
        evtstart.elapsed_time(evtend)

    @skip_on_cudasim("Testing cuda.core events requires driver")
    def test_event_query(self):
        stream = cuda.stream()
        evt = cuda.event()

        # Mapped arrays: host-side edits visible to device and vice versa.
        started = cuda.mapped_array(1, dtype=np.int32)
        release = cuda.mapped_array(1, dtype=np.int32)

        @cuda.jit
        def gated_kernel(started_flag, release_flag):
            # Signal that kernel has started
            started_flag[0] = 1
            # Spin until host releases us
            while release_flag[0] == 0:
                cuda.nanosleep(int32(1_000))

        # Compile first
        started[0] = 0
        release[0] = 1  # Don't block during warmup
        gated_kernel[1, 1, stream](started, release)
        stream.synchronize()

        # Reset for actual test
        started[0] = 0
        release[0] = 0

        # Launch - kernel will spin until we release it
        gated_kernel[1, 1, stream](started, release)
        evt.record(stream)

        # Wait until kernel confirms it's running
        while started[0] == 0:
            pass

        # Kernel is running until we release it - if query returns True, fail.
        immediate_query = evt.query()
        assert immediate_query is False, "Query returned True prematurely"

        # Release the kernel and synchronize
        release[0] = 1
        evt.synchronize()

        # If query returns False after synchronize, fail.
        synced_query = evt.query()
        assert synced_query is True, "Query returned False after sync"

if __name__ == "__main__":
    unittest.main()
