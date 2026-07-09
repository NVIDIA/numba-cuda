# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import sysconfig
import unittest

from numba.cuda.tests.support import run_in_subprocess


def sysconfig_var_is_true(name):
    value = sysconfig.get_config_var(name)
    return value is True or str(value).strip() == "1"


class TestImport(unittest.TestCase):
    def test_datetime_types(self):
        from numba import cuda, types

        self.assertIs(cuda.types.NPDatetime, types.NPDatetime)
        self.assertIs(cuda.types.NPTimedelta, types.NPTimedelta)

    def test_no_impl_import(self):
        """
        Tests that importing cuda doesn't trigger the import of modules
        containing lowering implementation that would likely install things in
        the builtins registry and have side effects impacting other targets.
        """

        banlist = (
            "numba.cpython.slicing",
            "numba.cpython.tupleobj",
            "numba.cpython.enumimpl",
            "numba.cpython.hashing",
            "numba.cpython.heapq",
            "numba.cpython.iterators",
            "numba.cpython.numbers",
            "numba.cpython.rangeobj",
            "numba.cpython.cmathimpl",
            "numba.cpython.mathimpl",
            "numba.cpython.printimpl",
            "numba.cuda.python.slicing",
            "numba.cuda.python.tupleobj",
            "numba.cuda.python.enumimpl",
            "numba.cuda.python.hashing",
            "numba.cuda.python.heapq",
            "numba.cuda.python.iterators",
            "numba.cuda.python.numbers",
            "numba.cuda.python.rangeobj",
            "numba.cuda.python.cmathimpl",
            "numba.cuda.python.mathimpl",
            "numba.cuda.python.printimpl",
            "numba.cuda.core.optional",
            "numba.cuda.misc.gdb_hook",
            "numba.cuda.misc.cffiimpl",
            "numba.np.linalg",
            "numba.np.polynomial",
            "numba.np.arraymath",
            "numba.np.npdatetime",
            "numba.np.npyimpl",
            "numba.cuda.np.linalg",
            "numba.cuda.np.polynomial",
            "numba.cuda.np.arraymath",
            "numba.cuda.np.npdatetime",
            "numba.cuda.np.npyimpl",
        )

        code = """\
import sys

from numba import cuda

for mod in sys.modules:
    print(mod)"""

        out, _ = run_in_subprocess(code)
        modlist = out.splitlines()
        unexpected = set(banlist) & set(modlist)
        assert not unexpected

    @unittest.skipUnless(
        sysconfig_var_is_true("Py_GIL_DISABLED"),
        "requires a free-threaded Python build",
    )
    def test_free_threaded_import_keeps_gil_disabled(self):
        code = """\
import importlib
import sys

assert hasattr(sys, "_is_gil_enabled")
assert not sys._is_gil_enabled()

import numba.cuda
assert not sys._is_gil_enabled()

for name in (
    "numba.cuda.cext._typeconv",
    "numba.cuda.cext.mviewbuf",
    "numba.cuda.cext._helperlib",
    "numba.cuda.cext._dispatcher",
):
    importlib.import_module(name)
    assert not sys._is_gil_enabled(), name
"""
        env = os.environ.copy()
        env["PYTHON_GIL"] = "0"
        run_in_subprocess(code, flags=("-W", "error::RuntimeWarning"), env=env)


if __name__ == "__main__":
    unittest.main()
