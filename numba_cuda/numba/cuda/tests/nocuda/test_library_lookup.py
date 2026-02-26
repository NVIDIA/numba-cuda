# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pathlib
import tempfile

from numba.cuda.testing import unittest
from numba.cuda.cuda_paths import (
    _find_cuda_home_from_lib_path,
    get_cuda_paths,
)


class TestGetCudaPaths(unittest.TestCase):
    def test_expected_keys(self):
        """get_cuda_paths() should return all expected component keys."""
        d = get_cuda_paths()
        expected_keys = {"nvrtc", "nvvm", "libdevice", "cudadevrt", "include_dir"}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_values_are_named_tuples(self):
        """Each value should be an _env_path_tuple with 'by' and 'info'."""
        d = get_cuda_paths()
        for key, val in d.items():
            self.assertTrue(
                hasattr(val, "by") and hasattr(val, "info"),
                f"{key} value missing 'by' or 'info' attributes",
            )

    def test_result_is_cached(self):
        """Repeated calls should return the same dict object."""
        d1 = get_cuda_paths()
        d2 = get_cuda_paths()
        self.assertIs(d1, d2)


class TestCudaHomeDetection(unittest.TestCase):
    def test_find_cuda_home(self):
        """Test the directory walking logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_root = pathlib.Path(tmpdir) / "cuda"
            lib64 = cuda_root / "lib64"
            nvvm = cuda_root / "nvvm"
            nvvm_lib64 = nvvm / "lib64"

            lib64.mkdir(parents=True)
            nvvm_lib64.mkdir(parents=True)

            nvrtc_path = lib64 / "libnvrtc.so.12"
            nvrtc_path.touch()

            nvvm_lib = nvvm_lib64 / "libnvvm.so.4"
            nvvm_lib.touch()

            found_cuda_home = _find_cuda_home_from_lib_path(str(nvrtc_path))

            expected = str(cuda_root.resolve())
            assert found_cuda_home == expected, (
                f"Expected {expected}, got {found_cuda_home}"
            )

            assert (pathlib.Path(found_cuda_home) / "nvvm").is_dir()

    def test_find_cuda_home_no_nvvm(self):
        """Walking up with no nvvm directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = pathlib.Path(tmpdir) / "lib64" / "libnvrtc.so.12"
            lib_path.parent.mkdir(parents=True)
            lib_path.touch()

            result = _find_cuda_home_from_lib_path(str(lib_path))
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
