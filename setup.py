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


setup(cmdclass={"build_py": build_py_with_redirector})
