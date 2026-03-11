# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import pickle
import tempfile
import unittest

import numba
import numba_cuda


class TestCacheVersion(unittest.TestCase):
    def test_cache_version_includes_numba_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
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
