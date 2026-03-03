# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import os
import pickle
import sys
import tempfile
import unittest

import numba
import numba_cuda


@contextlib.contextmanager
def _cuda_bindings_stub():
    tempdir = tempfile.TemporaryDirectory()
    try:
        cuda_root = os.path.join(tempdir.name, "cuda")
        bindings_root = os.path.join(cuda_root, "bindings")
        os.makedirs(bindings_root, exist_ok=True)
        for path in (cuda_root, bindings_root):
            init_path = os.path.join(path, "__init__.py")
            with open(init_path, "w", encoding="utf-8"):
                pass
        runtime_path = os.path.join(bindings_root, "runtime.py")
        with open(runtime_path, "w", encoding="utf-8") as f:
            f.write("def getLocalRuntimeVersion():\n    return (0, 0)\n")
        sys.path.insert(0, tempdir.name)
        yield
    finally:
        if tempdir.name in sys.path:
            sys.path.remove(tempdir.name)
        tempdir.cleanup()


class TestCacheVersion(unittest.TestCase):
    def test_cache_version_includes_numba_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _cuda_bindings_stub():
                from numba.cuda.core.caching import IndexDataCacheFile

                cache = IndexDataCacheFile(
                    cache_path=tmp,
                    filename_base="cache_version",
                    source_stamp=("stamp", 1),
                )
                cache.save(("sig",), ("payload",))

                index_path = os.path.join(tmp, "cache_version.nbi")
                with open(index_path, "rb") as f:
                    version = pickle.load(f)

        self.assertEqual(version, (numba.__version__, numba_cuda.__version__))

    def test_cache_version_mismatch_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _cuda_bindings_stub():
                from numba.cuda.core.caching import IndexDataCacheFile

                cache = IndexDataCacheFile(
                    cache_path=tmp,
                    filename_base="cache_version",
                    source_stamp=("stamp", 1),
                )
                index_path = os.path.join(tmp, "cache_version.nbi")
                with open(index_path, "wb") as f:
                    pickle.dump(("0.0.0", "0.0.0"), f, protocol=-1)

                self.assertEqual(cache._load_index(), {})


if __name__ == "__main__":
    unittest.main()
