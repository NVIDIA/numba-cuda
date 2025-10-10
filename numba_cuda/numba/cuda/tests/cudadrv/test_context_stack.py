# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numbers
from ctypes import byref
import weakref

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
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
                # Check that the device is an integer if not using the CUDA
                # Python bindings, otherwise it's a CUdevice object
                assert isinstance(device_id, int) != driver.USE_NV_BINDING
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
        if driver.USE_NV_BINDING:
            dev = driver.binding.CUdevice(0)
            binding_hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
            hctx = driver.drvapi.cu_context(int(binding_hctx))
        else:
            dev = 0
            hctx = driver.drvapi.cu_context()
            the_driver.cuDevicePrimaryCtxRetain(byref(hctx), dev)
        try:
            ctx = driver.Context(weakref.proxy(self), hctx)
            ctx.push()
            # Check that the context from numba matches the created primary
            # context.
            my_ctx = cuda.current_context()
            self.assertEqual(my_ctx.handle.value, ctx.handle.value)

            extra_work()
        finally:
            ctx.pop()
            the_driver.cuDevicePrimaryCtxRelease(dev)

    def test_attached_non_primary(self):
        # Emulate non-primary context creation by 3rd party
        the_driver = driver.driver
        if driver.USE_NV_BINDING:
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
        else:
            hctx = driver.drvapi.cu_context()
            the_driver.cuCtxCreate(byref(hctx), 0, 0)
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


if __name__ == "__main__":
    unittest.main()
