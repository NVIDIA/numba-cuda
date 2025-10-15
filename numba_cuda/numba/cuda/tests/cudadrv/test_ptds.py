# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import logging
import traceback
from numba.cuda.testing import unittest


def child_test():
    from numba import cuda, int32, void
    from numba.cuda.core import config
    import io
    import numpy as np
    import threading

    # Enable PTDS before we make any CUDA driver calls.  Enabling it first
    # ensures that PTDS APIs are used because the CUDA driver looks up API
    # functions on first use and memoizes them.
    config.CUDA_PER_THREAD_DEFAULT_STREAM = 1

    # Set up log capture for the Driver API so we can see what API calls were
    # used.
    logbuf = io.StringIO()
    handler = logging.StreamHandler(logbuf)
    cudadrv_logger = logging.getLogger("numba.cuda.cudadrv.driver")
    cudadrv_logger.addHandler(handler)
    cudadrv_logger.setLevel(logging.DEBUG)

    # Set up data for our test, and copy over to the device
    N = 2**16
    N_THREADS = 10
    N_ADDITIONS = 4096

    # Seed the RNG for repeatability
    np.random.seed(1)
    x = np.random.randint(low=0, high=1000, size=N, dtype=np.int32)
    r = np.zeros_like(x)

    # One input and output array for each thread
    xs = [cuda.to_device(x) for _ in range(N_THREADS)]
    rs = [cuda.to_device(r) for _ in range(N_THREADS)]

    # Compute the grid size and get the [per-thread] default stream
    n_threads = 256
    n_blocks = N // n_threads
    stream = cuda.default_stream()

    # A simple multiplication-by-addition kernel. What it does exactly is not
    # too important; only that we have a kernel that does something.
    @cuda.jit(void(int32[::1], int32[::1]))
    def f(r, x):
        i = cuda.grid(1)

        if i > len(r):
            return

        # Accumulate x into r
        for j in range(N_ADDITIONS):
            r[i] += x[i]

    # This function will be used to launch the kernel from each thread on its
    # own unique data.
    def kernel_thread(n):
        f[n_blocks, n_threads, stream](rs[n], xs[n])

    # Create threads
    threads = [
        threading.Thread(target=kernel_thread, args=(i,))
        for i in range(N_THREADS)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish, to ensure that we don't synchronize with
    # the device until all kernels are scheduled.
    for thread in threads:
        thread.join()

    # Synchronize with the device
    cuda.synchronize()

    # Check output is as expected
    expected = x * N_ADDITIONS
    for i in range(N_THREADS):
        np.testing.assert_equal(rs[i].copy_to_host(), expected)

    # Return the driver log output to the calling process for checking
    handler.flush()
    return logbuf.getvalue()


def child_test_wrapper(result_queue):
    try:
        output = child_test()
        success = True
    # Catch anything raised so it can be propagated
    except:  # noqa: E722
        output = traceback.format_exc()
        success = False

    result_queue.put((success, output))


if __name__ == "__main__":
    unittest.main()
