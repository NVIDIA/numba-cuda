# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import asyncio
import functools
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


def with_asyncio_loop(f):
    @functools.wraps(f)
    def runner(*args, **kwds):
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        try:
            return loop.run_until_complete(f(*args, **kwds))
        finally:
            loop.close()

    return runner


@unittest.skip("Disabled temporarily due to Issue #317")
@skip_on_cudasim("CUDA Driver API unsupported in the simulator")
class TestCudaStream(CUDATestCase):
    def test_add_callback(self):
        def callback(stream, status, event):
            event.set()

        stream = cuda.stream()
        callback_event = threading.Event()
        stream.add_callback(callback, callback_event)
        self.assertTrue(callback_event.wait(1.0))

    def test_add_callback_with_default_arg(self):
        callback_event = threading.Event()

        def callback(stream, status, arg):
            self.assertIsNone(arg)
            callback_event.set()

        stream = cuda.stream()
        stream.add_callback(callback)
        self.assertTrue(callback_event.wait(1.0))

    @with_asyncio_loop
    async def test_async_done(self):
        stream = cuda.stream()
        await stream.async_done()

    @with_asyncio_loop
    async def test_parallel_tasks(self):
        async def async_cuda_fn(value_in: float) -> float:
            stream = cuda.stream()
            h_src, h_dst = cuda.pinned_array(8), cuda.pinned_array(8)
            h_src[:] = value_in
            d_ary = cuda.to_device(h_src, stream=stream)
            d_ary.copy_to_host(h_dst, stream=stream)
            done_result = await stream.async_done()
            self.assertEqual(done_result, stream)
            return h_dst.mean()

        values_in = [1, 2, 3, 4]
        tasks = [asyncio.create_task(async_cuda_fn(v)) for v in values_in]
        values_out = await asyncio.gather(*tasks)
        self.assertTrue(np.allclose(values_in, values_out))

    @with_asyncio_loop
    async def test_multiple_async_done(self):
        stream = cuda.stream()
        done_aws = [stream.async_done() for _ in range(4)]
        done = await asyncio.gather(*done_aws)
        for d in done:
            self.assertEqual(d, stream)

    @with_asyncio_loop
    async def test_multiple_async_done_multiple_streams(self):
        streams = [cuda.stream() for _ in range(4)]
        done_aws = [stream.async_done() for stream in streams]
        done = await asyncio.gather(*done_aws)

        # Ensure we got the four original streams in done
        self.assertSetEqual(set(done), set(streams))

    @with_asyncio_loop
    async def test_cancelled_future(self):
        stream = cuda.stream()
        done1, done2 = stream.async_done(), stream.async_done()
        done1.cancel()
        await done2
        self.assertTrue(done1.cancelled())
        self.assertTrue(done2.done())


if __name__ == "__main__":
    unittest.main()
