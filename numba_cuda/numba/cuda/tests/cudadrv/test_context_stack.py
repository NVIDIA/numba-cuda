# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numbers
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from numba import cuda
from numba.cuda.testing import (
    unittest,
    CUDATestCase,
    ForeignArray,
    skip_on_cudasim,
)
from numba.cuda.cudadrv import driver


class TestContextStack(CUDATestCase):
    def setUp(self):
        super().setUp()
        # Reset before testing
        cuda.current_context().reset()

    def test_gpus_len(self):
        self.assertGreater(len(cuda.gpus), 0)

    def test_gpus_iter(self):
        gpulist = list(cuda.gpus)
        self.assertGreater(len(gpulist), 0)

    def test_gpus_cudevice_indexing(self):
        """Test that CUdevice objects can be used to index into cuda.gpus"""
        # When using the CUDA Python bindings, the device ids are CUdevice
        # objects, otherwise they are integers. We test that the device id is
        # usable as an index into cuda.gpus.
        device_ids = [device.id for device in cuda.list_devices()]
        for device_id in device_ids:
            with cuda.gpus[device_id]:
                self.assertEqual(cuda.gpus.current.id, device_id)


class TestContextAPI(CUDATestCase):
    def tearDown(self):
        super().tearDown()
        cuda.current_context().reset()

    def test_context_memory(self):
        try:
            mem = cuda.current_context().get_memory_info()
        except NotImplementedError:
            self.skipTest("EMM Plugin does not implement get_memory_info()")

        self.assertIsInstance(mem.free, numbers.Number)
        self.assertEqual(mem.free, mem[0])

        self.assertIsInstance(mem.total, numbers.Number)
        self.assertEqual(mem.total, mem[1])

        self.assertLessEqual(mem.free, mem.total)

    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
    @skip_on_cudasim("CUDA HW required")
    def test_forbidden_context_switch(self):
        # Cannot switch context inside a `cuda.require_context`
        @cuda.require_context
        def switch_gpu():
            with cuda.gpus[1]:
                pass

        with cuda.gpus[0]:
            with self.assertRaises(RuntimeError) as raises:
                switch_gpu()

            self.assertIn("Cannot switch CUDA-context.", str(raises.exception))

    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
    def test_accepted_context_switch(self):
        def switch_gpu():
            with cuda.gpus[1]:
                return cuda.current_context().device.id

        with cuda.gpus[0]:
            devid = switch_gpu()
        self.assertEqual(int(devid), 1)


@skip_on_cudasim("CUDA HW required")
class Test3rdPartyContext(CUDATestCase):
    def tearDown(self):
        super().tearDown()
        cuda.current_context().reset()

    def test_attached_primary(self, extra_work=lambda: None):
        # Emulate primary context creation by 3rd party
        the_driver = driver.driver
        dev = driver.binding.CUdevice(0)
        hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
        ctx = driver.Context(dev, hctx)
        try:
            ctx.push()
            # Check that the context from numba matches the created primary
            # context.
            my_ctx = cuda.current_context()
            self.assertEqual(int(my_ctx.handle), int(ctx.handle))

            extra_work()
        finally:
            ctx.pop()
            the_driver.cuDevicePrimaryCtxRelease(dev)

    def test_attached_non_primary(self):
        # Emulate non-primary context creation by 3rd party
        the_driver = driver.driver
        flags = 0
        dev = driver.binding.CUdevice(0)

        result, version = driver.binding.cuDriverGetVersion()
        self.assertEqual(
            result,
            driver.binding.CUresult.CUDA_SUCCESS,
            "Error getting CUDA driver version",
        )

        # CUDA 13's cuCtxCreate has an optional parameter prepended. The
        # version of cuCtxCreate in use depends on the cuda.bindings major
        # version rather than the installed driver version on the machine
        # we're running on.
        from cuda import bindings

        bindings_version = int(bindings.__version__.split(".")[0])
        if bindings_version in (11, 12):
            args = (flags, dev)
        else:
            args = (None, flags, dev)

        hctx = the_driver.cuCtxCreate(*args)
        try:
            cuda.current_context()
        except RuntimeError as e:
            # Expecting an error about non-primary CUDA context
            self.assertIn(
                "Numba cannot operate on non-primary CUDA context ", str(e)
            )
        else:
            self.fail("No RuntimeError raised")
        finally:
            the_driver.cuCtxDestroy(hctx)

    def test_cudajit_in_attached_primary_context(self):
        def do():
            from numba import cuda

            @cuda.jit
            def foo(a):
                for i in range(a.size):
                    a[i] = i

            a = cuda.device_array(10)
            foo[1, 1](a)
            self.assertEqual(list(a.copy_to_host()), list(range(10)))

        self.test_attached_primary(do)


@skip_on_cudasim("CUDA HW required")
class TestGreenContextInterop(CUDATestCase):
    def tearDown(self):
        super().tearDown()
        with driver.driver.get_active_context() as ac:
            if ac:
                cuda.current_context().reset()

    def _require_green_context_support(self):
        if driver.driver.get_version() < (13, 0):
            self.skipTest("CUDA 13+ required for green contexts")

        required = (
            "CUgreenCtxCreate_flags",
            "cuDeviceGetDevResource",
            "cuDevResourceGenerateDesc",
            "cuGreenCtxCreate",
            "cuCtxFromGreenCtx",
            "cuGreenCtxStreamCreate",
            "cuGreenCtxDestroy",
            "cuStreamGetGreenCtx",
        )
        missing = [
            name for name in required if not hasattr(driver.binding, name)
        ]
        if missing:
            self.skipTest(
                "Green context bindings are unavailable: "
                + ", ".join(sorted(missing))
            )

    @contextmanager
    def green_context(self):
        self._require_green_context_support()

        the_driver = driver.driver
        binding = driver.binding
        dev = binding.CUdevice(0)
        green_ctx = None
        ctx_handle = None
        stream = None

        try:
            resource = the_driver.cuDeviceGetDevResource(
                dev, binding.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
            desc = the_driver.cuDevResourceGenerateDesc([resource], 1)
            green_ctx = the_driver.cuGreenCtxCreate(
                desc,
                dev,
                binding.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM.value,
            )
            ctx_handle = the_driver.cuCtxFromGreenCtx(green_ctx)
            stream = the_driver.cuGreenCtxStreamCreate(
                green_ctx,
                binding.CUstream_flags.CU_STREAM_NON_BLOCKING.value,
                0,
            )
            the_driver.cuCtxPushCurrent(ctx_handle)
        except driver.CudaAPIError as e:
            if green_ctx is not None:
                the_driver.cuGreenCtxDestroy(green_ctx)
            self.skipTest(f"Green contexts are unavailable: {e}")

        try:
            yield ctx_handle, stream
        finally:
            if stream is not None:
                the_driver.cuStreamDestroy(stream)
            if ctx_handle is not None:
                popped = the_driver.cuCtxPopCurrent()
                self.assertEqual(int(popped), int(ctx_handle))
            if green_ctx is not None:
                the_driver.cuGreenCtxDestroy(green_ctx)

    def test_attached_green_context(self):
        with self.green_context() as (ctx_handle, _):
            my_ctx = cuda.current_context()
            self.assertEqual(int(my_ctx.handle), int(ctx_handle))
            self.assertTrue(my_ctx.borrowed)

    def test_cudajit_in_attached_green_context(self):
        with self.green_context() as (_, stream_handle):
            stream = cuda.external_stream(int(stream_handle))

            @cuda.jit
            def fill(a):
                i = cuda.grid(1)
                if i < a.size:
                    a[i] = i

            a = cuda.device_array(10, dtype=np.int32)
            fill[1, 10, stream](a)
            stream.synchronize()

            np.testing.assert_array_equal(
                a.copy_to_host(), np.arange(10, dtype=np.int32)
            )

    def test_cuda_array_interface_sync_in_green_context(self):
        with self.green_context() as (_, stream_handle):
            stream = cuda.external_stream(int(stream_handle))
            foreign = ForeignArray(
                cuda.device_array(10, dtype=np.int32, stream=stream)
            )

            @cuda.jit
            def touch(arr):
                i = cuda.grid(1)
                if i < arr.size:
                    arr[i] = i + 1

            with patch.object(
                cuda.cudadrv.driver.Stream, "synchronize", return_value=None
            ) as mock_sync:
                imported = cuda.as_cuda_array(foreign)

            self.assertTrue(imported.stream.external)
            self.assertEqual(int(imported.stream.handle), int(stream_handle))
            mock_sync.assert_called_once_with()

            with patch.object(
                cuda.cudadrv.driver.Stream, "synchronize", return_value=None
            ) as mock_sync:
                touch[1, 10](foreign)

            mock_sync.assert_called_once_with()

    def test_cufunc_cache_is_context_specific(self):
        from numba import types

        sig = (types.int32[::1],)

        @cuda.jit(sig)
        def fill(a):
            i = cuda.grid(1)
            if i < a.size:
                a[i] = i

        primary_ctx = cuda.current_context()
        primary = cuda.device_array(10, dtype=np.int32)
        fill[1, 10](primary)
        primary_key = primary_ctx.cache_key

        with self.green_context() as (_, _stream_handle):
            green_ctx = cuda.current_context()
            green = cuda.device_array(10, dtype=np.int32)
            fill[1, 10](green)
            np.testing.assert_array_equal(
                green.copy_to_host(), np.arange(10, dtype=np.int32)
            )
            green_key = green_ctx.cache_key

        fill[1, 10](primary)
        np.testing.assert_array_equal(
            primary.copy_to_host(), np.arange(10, dtype=np.int32)
        )

        cufunc_cache = fill.overloads[sig]._codelibrary._cufunc_cache
        self.assertIn(primary_key, cufunc_cache)
        self.assertIn(green_key, cufunc_cache)
        self.assertEqual(len(cufunc_cache), 2)

    def test_borrowed_context_reset_reloads_modules(self):
        with self.green_context():
            @cuda.jit
            def fill(a):
                i = cuda.grid(1)
                if i < a.size:
                    a[i] = i

            ctx = cuda.current_context()
            before = ctx.cache_key

            first = cuda.device_array(10, dtype=np.int32)
            fill[1, 10](first)
            np.testing.assert_array_equal(
                first.copy_to_host(), np.arange(10, dtype=np.int32)
            )
            overload = next(iter(fill.overloads.values()))

            ctx.reset()
            after = ctx.cache_key
            self.assertNotEqual(before, after)

            second = cuda.device_array(10, dtype=np.int32)
            fill[1, 10](second)
            np.testing.assert_array_equal(
                second.copy_to_host(), np.arange(10, dtype=np.int32)
            )

            cufunc_cache = overload._codelibrary._cufunc_cache
            self.assertIn(before, cufunc_cache)
            self.assertIn(after, cufunc_cache)

    def test_close_rejected_in_borrowed_context(self):
        with self.green_context():
            cuda.current_context()
            with self.assertRaises(RuntimeError) as raises:
                cuda.close()

            self.assertIn(
                "borrowed CUDA contexts are still live",
                str(raises.exception),
            )


if __name__ == "__main__":
    unittest.main()
