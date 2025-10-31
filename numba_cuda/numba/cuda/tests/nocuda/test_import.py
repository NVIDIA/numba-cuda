# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.tests.support import run_in_subprocess
import unittest


class TestImport(unittest.TestCase):
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

        code = "import sys; from numba import cuda; print(list(sys.modules))"

        out, _ = run_in_subprocess(code)
        modlist = set(eval(out.strip()))
        unexpected = set(banlist) & set(modlist)
        self.assertFalse(unexpected, "some modules unexpectedly imported")


if __name__ == "__main__":
    unittest.main()
