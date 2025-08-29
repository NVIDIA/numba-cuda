# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools
import os
import setuptools
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path


# Wire in distutils components from setuptools
CCompiler = setuptools.distutils.ccompiler.CCompiler
new_compiler = setuptools.distutils.ccompiler.new_compiler
customize_compiler = setuptools.distutils.sysconfig.customize_compiler
log = setuptools.distutils.log


@contextmanager
def _gentmpfile(suffix):
    # windows locks the tempfile so use a tempdir + file, see
    # https://github.com/numba/numba/issues/3304
    try:
        tmpdir = mkdtemp()
        ntf = open(os.path.join(tmpdir, "temp%s" % suffix), "wt")
        yield ntf
    finally:
        try:
            ntf.close()
            os.remove(ntf)
        except Exception:
            pass
        else:
            os.rmdir(tmpdir)


@functools.lru_cache(maxsize=1)
def external_compiler_works():
    """
    Returns True if the "external compiler" bound in numpy.distutil is present
    and working, False otherwise.
    """
    compiler = new_compiler()
    customize_compiler(compiler)
    for suffix in [".c", ".cxx"]:
        try:
            with _gentmpfile(suffix) as ntf:
                simple_c = "int main(void) { return 0; }"
                ntf.write(simple_c)
                ntf.flush()
                ntf.close()
                # *output_dir* is set to avoid the compiler putting temp files
                # in the current directory.
                compiler.compile([ntf.name], output_dir=Path(ntf.name).anchor)
        except Exception:  # likely CompileError or file system issue
            return False
    return True
