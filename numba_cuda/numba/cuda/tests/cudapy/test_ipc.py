# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import concurrent.futures
import multiprocessing as mp
import os
import itertools
import pickle

import numpy as np

from numba import cuda
from numba.cuda.testing import (
    skip_on_arm,
    skip_on_cudasim,
    skip_under_cuda_memcheck,
    skip_on_wsl2,
    CUDATestCase,
    ForeignArray,
)
from numba.cuda.tests.support import linux_only, windows_only
import unittest


def base_ipc_handle_test(handle, size, parent_pid):
    pid = os.getpid()
    assert pid != parent_pid
    dtype = np.dtype(np.intp)
    with cuda.open_ipc_array(
        handle, shape=size // dtype.itemsize, dtype=dtype
    ) as darr:
        # copy the data to host
        return darr.copy_to_host()


def serialize_ipc_handle_test(handle, parent_pid):
    pid = os.getpid()
    assert pid != parent_pid

    dtype = np.dtype(np.intp)
    darr = handle.open_array(
        cuda.current_context(),
        shape=handle.size // dtype.itemsize,
        dtype=dtype,
    )
    # copy the data to host
    arr = darr.copy_to_host()
    handle.close()
    return arr


def ipc_array_test(ipcarr, parent_pid):
    pid = os.getpid()
    assert pid != parent_pid
    with ipcarr as darr:
        arr = darr.copy_to_host()
        try:
            # should fail to reopen
            with ipcarr:
                pass
        except ValueError as e:
            if str(e) != "IpcHandle is already opened":
                raise AssertionError("invalid exception message")
        else:
            raise AssertionError("did not raise on reopen")
    return arr


class CUDAIpcTestCase(CUDATestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exe = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.exe.shutdown()
        del cls.exe


@linux_only
@skip_under_cuda_memcheck("Hangs cuda-memcheck")
@skip_on_cudasim("Ipc not available in CUDASIM")
@skip_on_arm("CUDA IPC not supported on ARM in Numba")
@skip_on_wsl2("CUDA IPC unreliable on WSL2; skipping IPC tests")
class TestIpcMemory(CUDAIpcTestCase):
    def test_ipc_handle(self):
        # prepare data for IPC
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)

        # create IPC handle
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)

        # manually prepare for serialization as bytes
        handle_bytes = ipch.handle.reserved
        size = ipch.size

        # spawn new process for testing
        fut = self.exe.submit(
            base_ipc_handle_test, handle_bytes, size, parent_pid=os.getpid()
        )
        out = fut.result(timeout=3)
        np.testing.assert_equal(arr, out)

    def variants(self):
        # Test with no slicing and various different slices
        indices = (None, slice(3, None), slice(3, 8), slice(None, 8))
        # Test with a Numba DeviceNDArray, or an array from elsewhere through
        # the CUDA Array Interface
        foreigns = (False, True)
        return itertools.product(indices, foreigns)

    def check_ipc_handle_serialization(self, index_arg=None, foreign=False):
        # prepare data for IPC
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        if index_arg is not None:
            devarr = devarr[index_arg]
        if foreign:
            devarr = cuda.as_cuda_array(ForeignArray(devarr))
        expect = devarr.copy_to_host()

        # create IPC handle
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)

        # pickle
        buf = pickle.dumps(ipch)
        ipch_recon = pickle.loads(buf)
        self.assertIs(ipch_recon.base, None)
        self.assertEqual(ipch_recon.size, ipch.size)

        self.assertEqual(ipch_recon.handle.reserved, ipch.handle.reserved)

        # spawn new process for testing
        fut = self.exe.submit(
            serialize_ipc_handle_test, ipch, parent_pid=os.getpid()
        )
        out = fut.result(timeout=3)
        np.testing.assert_equal(expect, out)

    def test_ipc_handle_serialization(self):
        for (
            index,
            foreign,
        ) in self.variants():
            with self.subTest(index=index, foreign=foreign):
                self.check_ipc_handle_serialization(index, foreign)

    def check_ipc_array(self, index_arg=None, foreign=False):
        # prepare data for IPC
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        # Slice
        if index_arg is not None:
            devarr = devarr[index_arg]
        if foreign:
            devarr = cuda.as_cuda_array(ForeignArray(devarr))
        expect = devarr.copy_to_host()
        ipch = devarr.get_ipc_handle()

        # spawn new process for testing
        fut = self.exe.submit(ipc_array_test, ipch, parent_pid=os.getpid())
        out = fut.result(timeout=3)
        np.testing.assert_equal(expect, out)

    def test_ipc_array(self):
        for (
            index,
            foreign,
        ) in self.variants():
            with self.subTest(index=index, foreign=foreign):
                self.check_ipc_array(index, foreign)


def staged_ipc_handle_test(handle, device_num, parent_pid):
    pid = os.getpid()
    assert pid != parent_pid
    with cuda.gpus[device_num]:
        this_ctx = cuda.devices.get_context()
        deviceptr = handle.open_staged(this_ctx)
        arrsize = handle.size // np.dtype(np.intp).itemsize
        hostarray = np.zeros(arrsize, dtype=np.intp)
        cuda.driver.device_to_host(
            hostarray,
            deviceptr,
            size=handle.size,
        )
        handle.close()
        return hostarray


def staged_ipc_array_test(ipcarr, device_num, parent_pid):
    pid = os.getpid()
    assert pid != parent_pid
    with cuda.gpus[device_num]:
        with ipcarr as darr:
            arr = darr.copy_to_host()
            try:
                # should fail to reopen
                with ipcarr:
                    pass
            except ValueError as e:
                if str(e) != "IpcHandle is already opened":
                    raise AssertionError("invalid exception message")
            else:
                raise AssertionError("did not raise on reopen")
    return arr


@linux_only
@skip_under_cuda_memcheck("Hangs cuda-memcheck")
@skip_on_cudasim("Ipc not available in CUDASIM")
@skip_on_arm("CUDA IPC not supported on ARM in Numba")
@skip_on_wsl2("CUDA IPC unreliable on WSL2; skipping IPC tests")
class TestIpcStaged(CUDAIpcTestCase):
    def test_staged(self):
        # prepare data for IPC
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        # create IPC handle
        ctx = cuda.current_context()
        ipch = ctx.get_ipc_handle(devarr.gpu_data)
        # pickle
        buf = pickle.dumps(ipch)
        ipch_recon = pickle.loads(buf)
        self.assertIs(ipch_recon.base, None)
        self.assertEqual(ipch_recon.handle.reserved, ipch.handle.reserved)
        self.assertEqual(ipch_recon.size, ipch.size)

        # Test on every CUDA devices
        ngpus = len(cuda.gpus)
        futures = [
            self.exe.submit(
                staged_ipc_handle_test, ipch, device_num, parent_pid=os.getpid()
            )
            for device_num in range(ngpus)
        ]

        for fut in concurrent.futures.as_completed(futures, timeout=3 * ngpus):
            np.testing.assert_equal(arr, fut.result())

    def test_ipc_array(self):
        for device_num in range(len(cuda.gpus)):
            # prepare data for IPC
            arr = np.random.random(10)
            devarr = cuda.to_device(arr)
            ipch = devarr.get_ipc_handle()

            # spawn new process for testing
            fut = self.exe.submit(
                staged_ipc_array_test, ipch, device_num, parent_pid=os.getpid()
            )
            out = fut.result(timeout=3)
            np.testing.assert_equal(arr, out)


@windows_only
@skip_on_cudasim("Ipc not available in CUDASIM")
class TestIpcNotSupported(CUDATestCase):
    def test_unsupported(self):
        arr = np.arange(10, dtype=np.intp)
        devarr = cuda.to_device(arr)
        with self.assertRaises(OSError) as raises:
            devarr.get_ipc_handle()
        errmsg = str(raises.exception)
        self.assertIn("OS does not support CUDA IPC", errmsg)


if __name__ == "__main__":
    unittest.main()
