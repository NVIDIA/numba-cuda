# Copyright (c) 2024, NVIDIA CORPORATION.
import pathlib

from setuptools import setup
from setuptools.command.build_py import build_py


# Adapted from https://stackoverflow.com/a/71137790
class build_py_with_redirector(build_py):  # noqa: N801
    """Include the redirector files in the generated wheel."""

    def copy_redirector_file(self, extension):
        redirector = f"_numba_cuda_redirector.{extension}"
        source = pathlib.Path("site-packages") / redirector
        destination = pathlib.Path(self.build_lib) / redirector
        self.copy_file(str(source), str(destination), preserve_mode=0)

    def run(self):
        super().run()
        self.copy_redirector_file("pth")
        self.copy_redirector_file("py")


setup(cmdclass={"build_py": build_py_with_redirector})
