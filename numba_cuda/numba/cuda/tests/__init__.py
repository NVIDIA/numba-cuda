from fnmatch import fnmatch
from numba.cuda.testing import ensure_supported_ccs_initialized
from numba.testing import unittest
from numba import cuda
from os.path import dirname, isfile, join, normpath, relpath, splitext

import os
import sys
import traceback


# Copied and modified from numba/testing/__init__.py, to handle the difference
# between the top dirs for Numba and the CUDA target
def load_testsuite(loader, dir):
    """Find tests in 'dir'."""
    top_level_dir = dirname(dirname(dirname(dirname(__file__))))
    try:
        suite = unittest.TestSuite()
        files = []
        for f in os.listdir(dir):
            path = join(dir, f)
            if isfile(path) and fnmatch(f, 'test_*.py'):
                files.append(f)
            elif isfile(join(path, '__init__.py')):
                suite.addTests(loader.discover(path,
                                               top_level_dir=top_level_dir))
        for f in files:
            # turn 'f' into a filename relative to the toplevel dir and
            # translate it to a module name. This differs from the
            # implementation in Numba, because the toplevel dir is the
            # numba_cuda module location, not the numba one.
            f = relpath(join(dir, f), top_level_dir)
            f = splitext(normpath(f.replace(os.path.sep, '.')))[0]
            suite.addTests(loader.loadTestsFromName(f))
        return suite
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    this_dir = dirname(__file__)
    ensure_supported_ccs_initialized()
    suite.addTests(load_testsuite(loader, join(this_dir, 'nocuda')))
    if cuda.is_available():
        suite.addTests(load_testsuite(loader, join(this_dir, 'cudasim')))
        gpus = cuda.list_devices()
        if gpus and gpus[0].compute_capability >= (2, 0):
            suite.addTests(load_testsuite(loader, join(this_dir, 'cudadrv')))
            suite.addTests(load_testsuite(loader, join(this_dir, 'cudapy')))
            suite.addTests(load_testsuite(loader, join(this_dir,
                                                       'doc_examples')))
        else:
            print("skipped CUDA tests because GPU CC < 2.0")
    else:
        print("skipped CUDA tests")
    return suite
