# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from concurrent.futures import ThreadPoolExecutor
import gc
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import unittest

import numpy as np

from numba import cuda
from numba.cuda import launchconfig, types
from numba.cuda.cext import _dispatcher, mviewbuf
from numba.cuda.core.errors import TypingError
from numba.cuda.core.rewrites import Rewrite, register_rewrite, rewrite_registry
from numba.cuda.cudadrv.driver import CudaAPIError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.cuda.tests.support import (
    fresh_struct_array,
    free_threading_stress_enabled,
    is_free_threaded_python,
    launch_subprocess_code,
    subprocess_marker_results,
    temp_directory,
)
from numba.cuda.typeconv import Conversion
from numba.cuda.typeconv.typeconv import TypeManager


skip_unless_free_threaded = unittest.skipUnless(
    is_free_threaded_python(), "requires a free-threaded Python build"
)
skip_unless_ft_stress = unittest.skipUnless(
    free_threading_stress_enabled(), "requires NUMBA_CUDA_FT_STRESS=1"
)


def _stress_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _stress_int_config(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    try:
        return int(raw), True
    except (TypeError, ValueError):
        return default, False


def _stress_int(name, default):
    return _stress_int_config(name, default)[0]


def _stress_seconds(default=30.0):
    return _stress_float("NUMBA_CUDA_FT_STRESS_SECONDS", default)


def _stress_workers(default=32):
    configured, explicit = _stress_int_config(
        "NUMBA_CUDA_FT_STRESS_WORKERS", default
    )
    if explicit:
        return max(1, configured)
    return max(1, min(configured, os.cpu_count() or 1))


def _stress_processes(default=8):
    configured, explicit = _stress_int_config(
        "NUMBA_CUDA_FT_STRESS_PROCESSES", default
    )
    if explicit:
        return max(1, configured)
    return max(1, min(configured, os.cpu_count() or 1))


def _stress_iters(default):
    return _stress_int("NUMBA_CUDA_FT_STRESS_ITERS", default)


_REWRITE_FLAG = "_numba_cuda_free_threading_stress_rewrite_registered"
_FT_STRESS_CACHE_RESULT = "__FT_STRESS_CACHE_RESULT__"

if free_threading_stress_enabled() and not getattr(
    rewrite_registry, _REWRITE_FLAG, False
):

    @register_rewrite("after-inference")
    class FreeThreadingStressRewrite(Rewrite):
        _TARGET_NAMES = {
            "ft_stress_lcs_kernel",
            "ft_stress_cached_lcs_kernel",
        }

        def __init__(self, state):
            super().__init__(state)
            self._block = None
            self._applied = False

        def match(self, func_ir, block, typemap, calltypes):
            if self._applied:
                return False
            if func_ir.func_id.func_name not in self._TARGET_NAMES:
                return False
            self._block = block
            return True

        def apply(self):
            cfg = launchconfig.ensure_current_launch_config()
            cfg.dispatcher.mark_launch_config_sensitive()
            self._applied = True
            return self._block

    setattr(rewrite_registry, _REWRITE_FLAG, True)


@cuda.jit
def ft_stress_plain_kernel(x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = x[i] + 1


@cuda.jit
def ft_stress_lcs_kernel(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


@cuda.jit
def ft_stress_bad_kernel(x):
    x[0] = undefined_global_name  # noqa: F821


@cuda.jit
def ft_stress_good_kernel(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


@skip_unless_ft_stress
@skip_unless_free_threaded
class TestFreeThreadingNoCudaStress(unittest.TestCase):
    def test_compute_fingerprint_stress(self):
        workers = _stress_workers(16)
        iters = _stress_iters(2000)
        stop = threading.Event()
        shared_list = [np.arange(4, dtype=np.int32)]
        shared_set = {1, 2, 3}

        def mutate_list():
            i = 0
            while not stop.is_set():
                shared_list[:] = [np.arange(4, dtype=np.int32)]
                shared_list.append(np.arange(2, dtype=np.float64))
                shared_list.pop()
                shared_list.clear()
                shared_list.append(np.arange((i % 3) + 1, dtype=np.int16))
                i += 1

        def mutate_set():
            i = 0
            while not stop.is_set():
                shared_set.clear()
                shared_set.add(i)
                shared_set.add(i + 1)
                shared_set.discard(i)
                i += 1

        def fingerprint(seed):
            rng = np.random.default_rng(seed)
            for _ in range(iters):
                i = int(rng.integers(0, 1000))
                values = (
                    np.arange(4, dtype=np.int32),
                    np.arange(6, dtype=np.float64).reshape(2, 3),
                    fresh_struct_array(i),
                    fresh_struct_array(i)[0],
                    (1, 2.0, np.float32(3), np.int64(4)),
                    [np.arange((i % 5) + 1, dtype=np.int16)],
                    np.float32(3.5),
                    np.complex128(1 + 2j),
                    {i},
                    shared_list,
                    shared_set,
                )
                for value in values:
                    try:
                        fp = _dispatcher.compute_fingerprint(value)
                    except ValueError as e:
                        if "empty" in str(e):
                            continue
                        raise
                    self.assertIsInstance(fp, bytes)

        mutators = [
            threading.Thread(target=mutate_list),
            threading.Thread(target=mutate_set),
        ]
        for mutator in mutators:
            mutator.start()
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(fingerprint, i) for i in range(workers)
                ]
                for future in futures:
                    future.result()
        finally:
            stop.set()
            for mutator in mutators:
                mutator.join(timeout=5)

        self.assertFalse(any(mutator.is_alive() for mutator in mutators))

    def test_mviewbuf_stress(self):
        workers = _stress_workers(16)
        iters = _stress_iters(20000)

        def check_info(n):
            for k in range(n):
                ndim = k % 4
                shape = tuple(range(1, ndim + 1)) if ndim else ()
                strides = tuple(4 for _ in range(ndim))
                start, end = mviewbuf.memoryview_get_extents_info(
                    shape, strides, ndim, 4
                )
                self.assertGreaterEqual(end, start)
                start, end = mviewbuf.memoryview_get_extents_info(
                    (3, 4), (16, 4), 2, 4
                )
                self.assertEqual(end - start, 48)

        def check_extents(n):
            for k in range(n):
                ary = np.arange((k % 8) + 1, dtype=np.float64)
                start, end = mviewbuf.memoryview_get_extents(ary)
                self.assertGreaterEqual(end, start)

        def check_bad_inputs(n):
            for _ in range(n):
                for args in (
                    ((3,), (4,), 2, 4),
                    ((3, 4), (16,), 2, 4),
                    ((3,), (4,), 1, 0),
                    ((3,), (4,), 1, -1),
                ):
                    with self.assertRaises(
                        (ValueError, OverflowError, TypeError)
                    ):
                        mviewbuf.memoryview_get_extents_info(*args)

        jobs = (check_info, check_extents, check_bad_inputs)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(jobs[i % len(jobs)], iters)
                for i in range(workers)
            ]
            for future in futures:
                future.result()

    def test_typeconv_stress(self):
        workers = _stress_workers(12)
        iters = _stress_iters(3000)
        manager = TypeManager()
        i16, i32, i64 = types.int16, types.int32, types.int64
        f32, f64 = types.float32, types.float64

        manager.set_promote(i32, i64)
        manager.set_unsafe_convert(i32, f32)
        manager.set_promote(i16, i32)
        manager.set_safe_convert(f32, f64)

        sig = (i32, f32)
        overloads = ((i32, i32), (f32, f32), (i64, i64), (i16, i16))

        def writer():
            for _ in range(iters):
                manager.set_promote(i32, i64)
                manager.set_unsafe_convert(i32, f32)
                manager.set_promote(i16, i32)
                manager.set_safe_convert(f32, f64)

        def reader():
            for _ in range(iters):
                self.assertEqual(
                    manager.check_compatible(i32, i64), Conversion.promote
                )
                self.assertEqual(
                    manager.check_compatible(i32, f32), Conversion.unsafe
                )
                self.assertEqual(
                    manager.check_compatible(i16, i32), Conversion.promote
                )
                self.assertEqual(
                    manager.check_compatible(f32, f64), Conversion.safe
                )
                manager.select_overload(sig, overloads, True, False)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(writer if i % 2 else reader)
                for i in range(workers)
            ]
            for future in futures:
                future.result()


@skip_on_cudasim("stress tests require the CUDA runtime")
@skip_unless_ft_stress
@skip_unless_free_threaded
class TestFreeThreadingCudaStress(CUDATestCase):
    def test_dispatch_lcs_callbacks_and_surfaces_stress(self):
        seconds = _stress_seconds(30.0)
        workers = _stress_workers(32)
        stop = threading.Event()
        errors = []
        errors_lock = threading.Lock()
        callback_hits = []
        callback_lock = threading.Lock()
        original_gc_threshold = gc.get_threshold()
        dtypes = (
            np.int16,
            np.int32,
            np.int64,
            np.uint16,
            np.float32,
            np.float64,
        )
        configs = [(b, t) for b in (1, 2, 4) for t in (32, 64, 128)]

        def record_error(where):
            with errors_lock:
                errors.append((where, traceback.format_exc()))
            stop.set()

        def callback(kernel, launch_config):
            with callback_lock:
                callback_hits.append((kernel, launch_config))

        callback_config = ft_stress_plain_kernel.configure(1, 64)
        callback_config.pre_launch_callbacks[:] = [callback]

        def launch_plain(seed):
            rng = np.random.default_rng(seed)
            try:
                while not stop.is_set():
                    dtype = dtypes[int(rng.integers(0, len(dtypes)))]
                    n = int(rng.integers(1, 200))
                    host = np.arange(n, dtype=dtype)
                    inp = cuda.to_device(host)
                    out = cuda.device_array(n, dtype=dtype)
                    blocks = (n + 63) // 64
                    ft_stress_plain_kernel[blocks, 64](inp, out)
                    np.testing.assert_array_equal(out.copy_to_host(), host + 1)
            except Exception:
                record_error("launch_plain")

        def launch_lcs(seed):
            rng = np.random.default_rng(seed + 1000)
            try:
                while not stop.is_set():
                    blocks, threads = configs[
                        int(rng.integers(0, len(configs)))
                    ]
                    n = blocks * threads
                    arr = cuda.to_device(np.zeros(n, dtype=np.int32))
                    ft_stress_lcs_kernel[blocks, threads](arr)
                    np.testing.assert_array_equal(
                        arr.copy_to_host(), np.ones(n, dtype=np.int32)
                    )
            except Exception:
                record_error("launch_lcs")

        def launch_fresh(seed):
            try:
                while not stop.is_set():

                    @cuda.jit
                    def kernel(x, y):
                        i = cuda.grid(1)
                        if i < x.size:
                            y[i] = x[i] * 2

                    host = np.arange(50, dtype=np.int32)
                    inp = cuda.to_device(host)
                    out = cuda.device_array(host.size, dtype=np.int32)
                    kernel[1, 64](inp, out)
                    np.testing.assert_array_equal(out.copy_to_host(), host * 2)
            except Exception:
                record_error("launch_fresh")

        def specialize_fresh(seed):
            try:
                while not stop.is_set():

                    @cuda.jit
                    def kernel(x):
                        i = cuda.grid(1)
                        if i < x.size:
                            x[i] += 1

                    host = np.zeros(32, dtype=np.int32)
                    specialized = kernel.specialize(host)
                    arr = cuda.to_device(host)
                    specialized[1, 32](arr)
            except Exception:
                record_error("specialize_fresh")

        def fingerprint(seed):
            shared = [np.arange(4, dtype=np.int32)]

            def mutate():
                i = 0
                while not stop.is_set():
                    shared[:] = [np.arange((i % 4) + 1, dtype=np.int16)]
                    i += 1

            mutator = threading.Thread(target=mutate)
            mutator.start()
            try:
                while not stop.is_set():
                    values = (
                        shared,
                        (1, 2.0, np.float32(3)),
                        np.zeros(3, np.float64),
                    )
                    for value in values:
                        try:
                            _dispatcher.compute_fingerprint(value)
                        except ValueError as e:
                            if "empty" not in str(e):
                                raise
            except Exception:
                record_error("fingerprint")
            finally:
                mutator.join(timeout=5)

        def streams_events(seed):
            try:
                while not stop.is_set():
                    stream = cuda.stream()
                    event = cuda.event()
                    host = np.arange(64, dtype=np.float32)
                    arr = cuda.to_device(host, stream=stream)
                    event.record(stream=stream)
                    stream.synchronize()
                    del arr, event, stream
            except Exception:
                record_error("streams_events")

        def pinned(seed):
            try:
                while not stop.is_set():
                    host = np.full(64, 7, dtype=np.int32)
                    with cuda.pinned(host):
                        arr = cuda.to_device(host)
                        np.testing.assert_array_equal(arr.copy_to_host(), host)
                    pinned_arr = cuda.pinned_array(32, dtype=np.float32)
                    pinned_arr[:] = 1.5
            except Exception:
                record_error("pinned")

        def managed(seed):
            try:
                while not stop.is_set():
                    arr = cuda.managed_array(32, dtype=np.float32)
                    arr[:] = 0
                    ft_stress_good_kernel[1, 32](arr)
                    cuda.synchronize()
            except Exception:
                record_error("managed")

        def compile_failures(seed):
            try:
                while not stop.is_set():
                    with self.assertRaises(TypingError):
                        ft_stress_bad_kernel[1, 1](np.zeros(1, dtype=np.int32))
                    arr = cuda.to_device(np.zeros(8, dtype=np.int32))
                    ft_stress_good_kernel[1, 8](arr)
                    np.testing.assert_array_equal(
                        arr.copy_to_host(), np.ones(8, dtype=np.int32)
                    )
            except Exception:
                record_error("compile_failures")

        def mutate_callbacks(seed):
            try:
                while not stop.is_set():
                    callback_config.pre_launch_callbacks.append(callback)
                    if len(callback_config.pre_launch_callbacks) > 50:
                        del callback_config.pre_launch_callbacks[:]
                    time.sleep(0)
            except Exception:
                record_error("mutate_callbacks")

        def collect_gc(seed):
            try:
                while not stop.is_set():
                    gc.collect()
            except Exception:
                record_error("collect_gc")

        cuda.to_device(np.zeros(1, dtype=np.float32)).copy_to_host()
        managed_supported = True
        try:
            managed_probe = cuda.managed_array(1, dtype=np.float32)
            del managed_probe
        except CudaAPIError as e:
            message = str(e).lower()
            if "not supported" in message or "not_supported" in message:
                managed_supported = False
            else:
                raise

        jobs = (
            launch_plain,
            launch_lcs,
            launch_fresh,
            specialize_fresh,
            fingerprint,
            streams_events,
            pinned,
            compile_failures,
            mutate_callbacks,
            collect_gc,
        )
        if managed_supported:
            jobs = jobs[:7] + (managed,) + jobs[7:]
        selected_jobs = [jobs[i % len(jobs)] for i in range(workers)]
        deadline = time.monotonic() + seconds
        gc.set_threshold(50, 5, 5)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(job, i)
                    for i, job in enumerate(selected_jobs)
                ]
                while time.monotonic() < deadline and not errors:
                    time.sleep(0.2)
                stop.set()
                for future in futures:
                    future.result()
        finally:
            del callback_config.pre_launch_callbacks[:]
            gc.set_threshold(*original_gc_threshold)

        self.assertEqual(errors, [])
        if not callback_hits:
            callback_config.pre_launch_callbacks[:] = [callback]
            try:
                arr = cuda.to_device(np.zeros(64, dtype=np.int32))
                callback_config(arr, arr)
            finally:
                del callback_config.pre_launch_callbacks[:]
        self.assertGreater(len(callback_hits), 0)

    def test_launch_config_sensitive_disk_cache_stress(self):
        workers = _stress_workers(12)
        processes = _stress_processes(8)
        tempdir = temp_directory("ft_stress_cache")
        modname = "ft_stress_lcs_cache_fodder"
        here = os.path.dirname(__file__)
        usecase = os.path.join(
            here, "..", "cudapy", "cache_launch_config_sensitive_usecases.py"
        )
        modfile = os.path.join(tempdir, modname + ".py")
        cache_dir = os.path.join(tempdir, "__pycache__")
        shutil.copy(usecase, modfile)
        os.chmod(modfile, stat.S_IREAD | stat.S_IWRITE)

        def cache_files():
            files = []
            for root, _dirs, filenames in os.walk(cache_dir):
                files.extend(os.path.join(root, f) for f in filenames)
            return files

        def assert_cache_artifacts():
            files = cache_files()
            self.assertTrue(any(f.endswith(".lcs") for f in files), files)
            self.assertTrue(any(f.endswith(".nbi") for f in files), files)

        sys.path.insert(0, tempdir)
        try:
            mod = __import__(modname)
            blockdims = [32, 64] * max(1, workers // 2)
            barrier = threading.Barrier(len(blockdims))

            def thread_worker(blockdim):
                barrier.wait(timeout=30)
                return int(mod.launch(blockdim)[0])

            with ThreadPoolExecutor(max_workers=len(blockdims)) as executor:
                futures = [
                    executor.submit(thread_worker, blockdim)
                    for blockdim in blockdims
                ]
                results = [future.result(timeout=120) for future in futures]
            self.assertEqual(results, [1] * len(blockdims))
            assert_cache_artifacts()

            shutil.rmtree(cache_dir, ignore_errors=True)
            blockdims = [32, 64] * max(1, processes // 2)
            process_list = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        launch_subprocess_code(
                            tempdir,
                            modname,
                            blockdim,
                            _FT_STRESS_CACHE_RESULT,
                        ),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for blockdim in blockdims
            ]
            results = subprocess_marker_results(
                process_list,
                _FT_STRESS_CACHE_RESULT,
                timeout=180,
            )
            self.assertEqual(results, [1] * len(blockdims))
            assert_cache_artifacts()
        finally:
            sys.path.remove(tempdir)
            sys.modules.pop(modname, None)


if __name__ == "__main__":
    unittest.main()
