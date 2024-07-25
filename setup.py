# Copyright (c) 2024, NVIDIA CORPORATION.
import pathlib

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel, _TopLevelFinder

REDIRECTOR_PTH = "_numba_cuda_redirector.pth"
REDIRECTOR_PY = "_numba_cuda_redirector.py"
SITE_PACKAGES = pathlib.Path("site-packages")


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
        src.extend([
            str(SITE_PACKAGES / REDIRECTOR_PTH),
            str(SITE_PACKAGES / REDIRECTOR_PY),
        ])
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
            raise RuntimeError("Only lenient mode is supported for editable "
                               f"install. Current mode is {self.mode}")

        return TopLevelFinderWithRedirector(self.distribution, name)


setup(cmdclass={"build_py": build_py_with_redirector,
                "editable_wheel": editable_wheel_with_redirector})
