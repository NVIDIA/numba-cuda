# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import pickle
import tempfile
import unittest
from unittest import mock

import numba
import numba_cuda

from numba.cuda.core.caching import Cache, IndexDataCacheFile


def dummy_func(x):
    return x


class DummyCodegen:
    def magic_tuple(self):
        return ("dummy",)


class TestCacheVersion(unittest.TestCase):
    @staticmethod
    def make_key(version=None):
        cache = object.__new__(Cache)
        cache._py_func = dummy_func
        codegen = DummyCodegen()
        if version is None:
            return Cache._index_key(cache, ("sig",), codegen)
        with mock.patch.object(numba_cuda, "__version__", version):
            return Cache._index_key(cache, ("sig",), codegen)

    def test_cache_index_version_matches_numba(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = IndexDataCacheFile(
                cache_path=tmp,
                filename_base="cache_version",
                source_stamp=("stamp", 1),
            )
            cache.save(self.make_key(), ("payload",))

            index_path = os.path.join(tmp, "cache_version.nbi")
            with open(index_path, "rb") as f:
                version = pickle.load(f)

        self.assertEqual(version, numba.__version__)

    def test_cache_key_includes_numba_cuda(self):
        key = self.make_key()

        self.assertEqual(key[2], numba_cuda.__version__)

    def test_cache_key_version_mismatch_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = IndexDataCacheFile(
                cache_path=tmp,
                filename_base="cache_version",
                source_stamp=("stamp", 1),
            )
            old_key = self.make_key("0.0.0")
            new_key = self.make_key("9.9.9")
            cache.save(old_key, ("payload",))

            self.assertEqual(cache.load(old_key), ("payload",))
            self.assertIsNone(cache.load(new_key))


if __name__ == "__main__":
    unittest.main()
