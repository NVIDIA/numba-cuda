# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pathlib
import sys

from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel, _TopLevelFinder
from setuptools.command.build_ext import build_ext

REDIRECTOR_PTH = "_numba_cuda_redirector.pth"
REDIRECTOR_PY = "_numba_cuda_redirector.py"
SITE_PACKAGES = pathlib.Path("site-packages")


def get_version():
    """Read version from VERSION file."""
    version_file = pathlib.Path(__file__).parent / "numba_cuda" / "VERSION"
    return version_file.read_text().strip()


def get_ext_modules():
    """
    Return a list of Extension instances for the setup() call.
    """
    # Note we don't import NumPy at the toplevel, since setup.py
    # should be able to run without NumPy for pip to discover the
    # build dependencies. Need NumPy headers and libm linkage.
    import numpy as np

    np_compile_args = {
        "include_dirs": [
            np.get_include(),
        ],
    }
    if sys.platform != "win32":
        np_compile_args["libraries"] = [
            "m",
        ]

    ext_devicearray = Extension(
        name="numba_cuda.numba.cuda.cext._devicearray",
        sources=["numba_cuda/numba/cuda/cext/_devicearray.cpp"],
        depends=[
            "numba_cuda/numba/cuda/cext/_pymodule.h",
            "numba_cuda/numba/cuda/cext/_devicearray.h",
        ],
        include_dirs=["numba_cuda/numba/cuda/cext"],
        extra_compile_args=["-std=c++11"],
    )

    install_name_tool_fixer = []
    if sys.platform == "darwin":
        install_name_tool_fixer = ["-headerpad_max_install_names"]

    ext_mviewbuf = Extension(
        name="numba_cuda.numba.cuda.cext.mviewbuf",
        extra_link_args=install_name_tool_fixer,
        sources=["numba_cuda/numba/cuda/cext/mviewbuf.c"],
    )

    dispatcher_sources = [
        "numba_cuda/numba/cuda/cext/_dispatcher.cpp",
        "numba_cuda/numba/cuda/cext/_typeof.cpp",
        "numba_cuda/numba/cuda/cext/_hashtable.cpp",
        "numba_cuda/numba/cuda/cext/typeconv.cpp",
    ]
    ext_dispatcher = Extension(
        name="numba_cuda.numba.cuda.cext._dispatcher",
        sources=dispatcher_sources,
        depends=[
            "numba_cuda/numba/cuda/cext/_pymodule.h",
            "numba_cuda/numba/cuda/cext/_typeof.h",
            "numba_cuda/numba/cuda/cext/_hashtable.h",
        ],
        extra_compile_args=["-std=c++11"],
        **np_compile_args,
    )

    ext_typeconv = Extension(
        name="numba_cuda.numba.cuda.cext._typeconv",
        sources=[
            "numba_cuda/numba/cuda/cext/typeconv.cpp",
            "numba_cuda/numba/cuda/cext/_typeconv.cpp",
        ],
        depends=["numba_cuda/numba/cuda/cext/_pymodule.h"],
        extra_compile_args=["-std=c++11"],
    )

    # Append our cext dir to include_dirs
    ext_dispatcher.include_dirs.append("numba_cuda/numba/cuda/cext")

    return [ext_dispatcher, ext_typeconv, ext_mviewbuf, ext_devicearray]


def is_building():
    """
    Parse the setup.py command and return whether a build is requested.
    If False is returned, only an informational command is run.
    If True is returned, information about C extensions will have to
    be passed to the setup() function.
    """
    if len(sys.argv) < 2:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    build_commands = [
        "build",
        "build_py",
        "build_ext",
        "build_clib",
        "build_scripts",
        "install",
        "install_lib",
        "install_headers",
        "install_scripts",
        "install_data",
        "sdist",
        "bdist",
        "bdist_dumb",
        "bdist_rpm",
        "bdist_wininst",
        "check",
        "build_docs",
        "bdist_wheel",
        "bdist_egg",
        "develop",
        "easy_install",
        "test",
        "editable_wheel",
    ]
    return any(bc in sys.argv[1:] for bc in build_commands)


# Adapted from https://stackoverflow.com/a/71137790
class build_py_with_redirector(build_py):  # noqa: N801
    """Include the redirector files in the generated wheel."""

    def copy_redirector_file(self, source, destination="."):
        destination = pathlib.Path(self.build_lib) / destination
        self.copy_file(str(source), str(destination), preserve_mode=0)

    def run(self):
        super().run()
        self.copy_redirector_file(SITE_PACKAGES / REDIRECTOR_PTH)
        self.copy_redirector_file(SITE_PACKAGES / REDIRECTOR_PY)

    def get_source_files(self):
        src = super().get_source_files()
        src.extend(
            [
                str(SITE_PACKAGES / REDIRECTOR_PTH),
                str(SITE_PACKAGES / REDIRECTOR_PY),
            ]
        )
        return src

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        build_lib = pathlib.Path(self.build_lib)
        mapping[str(build_lib / REDIRECTOR_PTH)] = REDIRECTOR_PTH
        mapping[str(build_lib / REDIRECTOR_PY)] = REDIRECTOR_PY
        return mapping


class TopLevelFinderWithRedirector(_TopLevelFinder):
    """Include the redirector files in the editable wheel."""

    def get_implementation(self):
        for item in super().get_implementation():
            yield item

        with open(SITE_PACKAGES / REDIRECTOR_PTH) as f:
            yield (REDIRECTOR_PTH, f.read())

        with open(SITE_PACKAGES / REDIRECTOR_PY) as f:
            yield (REDIRECTOR_PY, f.read())


class editable_wheel_with_redirector(editable_wheel):
    def _select_strategy(self, name, tag, build_lib):
        # The default mode is "lenient" - others are "strict" and "compat".
        # "compat" is deprecated. "strict" creates a tree of links to files in
        # the repo. It could be implemented, but we only handle the default
        # case for now.
        if self.mode is not None and self.mode != "lenient":
            raise RuntimeError(
                "Only lenient mode is supported for editable "
                f"install. Current mode is {self.mode}"
            )

        return TopLevelFinderWithRedirector(self.distribution, name)


cmdclass = {}

numba_be_user_options = [
    ("werror", None, "Build extensions with -Werror"),
    ("wall", None, "Build extensions with -Wall"),
    ("noopt", None, "Build extensions without optimization"),
]


class NumbaBuildExt(build_ext):
    user_options = build_ext.user_options + numba_be_user_options
    boolean_options = build_ext.boolean_options + ["werror", "wall", "noopt"]

    def initialize_options(self):
        super().initialize_options()
        self.werror = 0
        self.wall = 0
        self.noopt = 0

    def run(self):
        extra_compile_args = []
        if self.noopt:
            if sys.platform == "win32":
                extra_compile_args.append("/Od")
            else:
                extra_compile_args.append("-O0")
        if self.werror:
            extra_compile_args.append("-Werror")
        if self.wall:
            extra_compile_args.append("-Wall")
        for ext in self.extensions:
            ext.extra_compile_args.extend(extra_compile_args)

        super().run()


cmdclass["build_ext"] = NumbaBuildExt
cmdclass["build_py"] = build_py_with_redirector
cmdclass["editable_wheel"] = editable_wheel_with_redirector

if is_building():
    ext_modules = get_ext_modules()
else:
    ext_modules = []

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
