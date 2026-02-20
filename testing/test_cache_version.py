# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import contextlib
import importlib.util
import os
import pickle
import sys
import tempfile
import types
import unittest

import numba_cuda


STUB_NUMBA_VERSION = "0.0.test"


@contextlib.contextmanager
def _stub_cuda_imports():
    def make_module(name):
        return types.ModuleType(name)

    stubs = {}
    existing = {}
    modules = [
        "numba",
        "numba.cuda",
        "numba.cuda.core",
        "numba.cuda.core.config",
        "numba.cuda.misc",
        "numba.cuda.misc.appdirs",
        "numba.cuda.serialize",
    ]

    for name in modules:
        if name in sys.modules:
            existing[name] = sys.modules[name]

    try:
        for name in modules:
            mod = make_module(name)
            stubs[name] = mod
            sys.modules[name] = mod

        numba_mod = sys.modules["numba"]
        numba_mod.__version__ = STUB_NUMBA_VERSION

        config_mod = sys.modules["numba.cuda.core.config"]
        config_mod.DEBUG_CACHE = False
        config_mod.CACHE_DIR = ""

        appdirs_mod = sys.modules["numba.cuda.misc.appdirs"]

        class AppDirs:
            def __init__(self, appname, appauthor=False):
                self.user_cache_dir = tempfile.gettempdir()

        appdirs_mod.AppDirs = AppDirs

        serialize_mod = sys.modules["numba.cuda.serialize"]
        serialize_mod.dumps = pickle.dumps

        yield
    finally:
        for name in modules:
            if name in existing:
                sys.modules[name] = existing[name]
            else:
                sys.modules.pop(name, None)


def _load_caching_module():
    caching_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "numba_cuda",
        "numba",
        "cuda",
        "core",
        "caching.py",
    )
    caching_path = os.path.abspath(caching_path)
    spec = importlib.util.spec_from_file_location(
        "numba_cuda_cache_under_test", caching_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCacheVersion(unittest.TestCase):
    def test_cache_version_includes_numba_cuda(self):
        with _stub_cuda_imports():
            caching = _load_caching_module()
            IndexDataCacheFile = caching.IndexDataCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            cache = IndexDataCacheFile(
                cache_path=tmp,
                filename_base="cache_version",
                source_stamp=("stamp", 1),
            )
            cache.save(("sig",), ("payload",))

            index_path = os.path.join(tmp, "cache_version.nbi")
            with open(index_path, "rb") as f:
                version = pickle.load(f)

        self.assertEqual(
            version, (STUB_NUMBA_VERSION, numba_cuda.__version__)
        )

    def test_cache_version_mismatch_invalidates(self):
        with _stub_cuda_imports():
            caching = _load_caching_module()
            IndexDataCacheFile = caching.IndexDataCacheFile

        with tempfile.TemporaryDirectory() as tmp:
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
