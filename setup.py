# Copyright (c) 2024, NVIDIA CORPORATION.
import pathlib

from setuptools import setup
from setuptools.command.build_py import build_py


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


setup(cmdclass={"build_py": build_py_with_redirector})
