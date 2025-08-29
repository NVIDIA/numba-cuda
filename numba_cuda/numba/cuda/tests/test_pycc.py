# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest
import sys
import os
from numba.cuda.tests.support import TestCase, needs_setuptools


_skip_reason = "windows only"
_windows_only = unittest.skipIf(
    not sys.platform.startswith("win"), _skip_reason
)


@needs_setuptools
class TestCompilerChecks(TestCase):
    # NOTE: THIS TEST MUST ALWAYS RUN ON WINDOWS, DO NOT SKIP
    @_windows_only
    def test_windows_compiler_validity(self):
        # When inside conda-build VSINSTALLDIR should be set and windows should
        # have a valid compiler available, `external_compiler_works()` should
        # agree with this. If this is not the case then error out to alert devs.

        # This is a local import to avoid deprecation warnings being generated
        # through the use of the numba.pycc module.
        from numba.pycc.platform import external_compiler_works

        is_running_conda_build = os.environ.get("CONDA_BUILD", None) is not None
        if is_running_conda_build:
            if os.environ.get("VSINSTALLDIR", None) is not None:
                self.assertTrue(external_compiler_works())
