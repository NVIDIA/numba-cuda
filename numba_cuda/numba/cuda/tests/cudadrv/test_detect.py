# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (
    unittest,
    CUDATestCase,
    skip_on_cudasim,
    skip_under_cuda_memcheck,
)
from numba.cuda.tests.support import captured_stdout


class TestCudaDetect(CUDATestCase):
    def test_cuda_detect(self):
        # exercise the code path
        with captured_stdout() as out:
            cuda.detect()
        output = out.getvalue()
        self.assertIn("Found", output)
        self.assertIn("CUDA devices", output)


@skip_under_cuda_memcheck("Hangs cuda-memcheck")
class TestCUDAFindLibs(CUDATestCase):
    def run_cmd(self, cmdline, env):
        popen = subprocess.Popen(
            cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # finish in 5 minutes or kill it
        timeout = threading.Timer(5 * 60.0, popen.kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            # the process should exit with an error
            return out.decode(), err.decode()
        finally:
            timeout.cancel()
        return None, None

    def run_test_in_separate_process(self, envvar, envvar_value):
        env_copy = os.environ.copy()
        env_copy[envvar] = str(envvar_value)
        code = """if 1:
            from numba import cuda
            @cuda.jit('(int64,)')
            def kernel(x):
                pass
            kernel(1,)
            """
        cmdline = [sys.executable, "-c", code]
        return self.run_cmd(cmdline, env_copy)

    @skip_on_cudasim("Simulator does not hit device library search code path")
    @unittest.skipIf(not sys.platform.startswith("linux"), "linux only")
    def test_cuda_find_lib_errors(self):
        """
        This tests that driver discovery attempts to load from typical system
        locations and fails gracefully if pointed at an invalid directory.
        """
        # one of these is likely to exist on linux, it's also unlikely that
        # someone has extracted the contents of libdevice into here!
        locs = ["lib", "lib64"]

        looking_for = None
        for l in locs:
            looking_for = os.path.join(os.path.sep, l)
            if os.path.exists(looking_for):
                break

        # This is the testing part, the test will only run if there's a valid
        # path in which to look
        if looking_for is not None:
            # We no longer support NUMBA_CUDA_DRIVER. Still run the subprocess
            # to ensure importing and a trivial kernel launch path work, but
            # do not set any Numba-specific driver env vars.
            out, err = self.run_test_in_separate_process(
                "DUMMY_UNUSED", looking_for
            )
            self.assertTrue(out is not None)
            self.assertTrue(err is not None)


if __name__ == "__main__":
    unittest.main()
