# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda, int32
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda._compat import Device
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

        @cuda.jit
        def spin(ms):
            # Sleep for ms
            for i in range(ms):
                cuda.nanosleep(int32(1_000_000))  # 1 ms

        # Run once to compile
        spin[1, 1, stream](1)

        spin_ms = 200
        spin[1, 1, stream](spin_ms)
        evt.record(stream)

        immediate_query = evt.query()
        evt.synchronize()
        synced_query = evt.query()

        assert immediate_query is False, "Query returned True prematurely"
        assert synced_query is True, "Query returned False after sync"

if __name__ == "__main__":
    unittest.main()
