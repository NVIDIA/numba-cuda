# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba import cuda
from numba.cuda import config
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda._compat import Device
from numba.cuda.testing import skip_on_cudasim, skip_if_cupy_unavailable

if config.ENABLE_CUDASIM:
    import numpy as cp
else:
    try:
        import cupy as cp
    except ImportError:
        cp = None


class TestCudaEvent(CUDATestCase):
    @skip_if_cupy_unavailable
    def test_event_elapsed(self):
        N = 32
        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record()
        dary = cp.array(np.arange(N, dtype=np.double))  # noqa: F841
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
        @cuda.jit
        def kernel():
            pass

        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record(stream=stream)

        kernel[1, 1, stream]()

        evtend.record(stream=stream)
        evtend.wait(stream=stream)
        evtend.synchronize()
        # Exercise the code path
        evtstart.elapsed_time(evtend)


if __name__ == "__main__":
    unittest.main()
