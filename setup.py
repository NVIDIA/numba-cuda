# Copyright (c) 2024, NVIDIA CORPORATION.
import logging
import pathlib

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel, _TopLevelFinder


# Adapted from https://stackoverflow.com/a/71137790
class build_py_with_redirector(build_py):  # noqa: N801
    """Include the redirector files in the generated wheel."""

    def copy_redirector_file(self, source, destination="."):
        destination = pathlib.Path(self.build_lib) / destination
        self.copy_file(str(source), str(destination), preserve_mode=0)

    def run(self):
        super().run()
        site_packages = pathlib.Path("site-packages")
        self.copy_redirector_file(site_packages / "_numba_cuda_redirector.pth")
        self.copy_redirector_file(site_packages / "_numba_cuda_redirector.py")

    def get_source_files(self):
        src = super().get_source_files()
        site_packages = pathlib.Path("site-packages")
        src.extend([
            str(site_packages / "_numba_cuda_redirector.pth"),
            str(site_packages / "_numba_cuda_redirector.py"),
        ])
        return src

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        build_lib = pathlib.Path(self.build_lib)
        mapping[str(build_lib / "_numba_cuda_redirector.pth")] = \
            "_numba_cuda_redirector.pth"
        mapping[str(build_lib / "_numba_cuda_redirector.py")] = \
            "_numba_cuda_redirector.py"
        return mapping


class TopLevelFinderWithRedirector(_TopLevelFinder):
    """Include the redirector files in the editable wheel."""

    def get_implementation(self):
        for item in super().get_implementation():
            yield item

        site_packages = pathlib.Path("site-packages")
        pth_file = "_numba_cuda_redirector.pth"
        py_file = "_numba_cuda_redirector.py"

        with open(site_packages / pth_file) as f:
            yield (pth_file, f.read())

        with open(site_packages / py_file) as f:
            yield (py_file, f.read())


class editable_wheel_with_redirector(editable_wheel):
    def _select_strategy(self, name, tag, build_lib):
        # The default mode is "lenient" - others are "strict" and "compat".
        # "compat" is deprecated. "strict" creates a tree of links to files in
        # the repo. It could be implemented, but we only handle the default
        # case for now.
        if self.mode is not None and self.mode != "lenient":
            raise RuntimeError("Only lenient mode is supported for editable "
                               f"install. Current mode is {self.mode}")

        return TopLevelFinderWithRedirector(self.distribution, name)


setup(cmdclass={"build_py": build_py_with_redirector,
                "editable_wheel": editable_wheel_with_redirector})
