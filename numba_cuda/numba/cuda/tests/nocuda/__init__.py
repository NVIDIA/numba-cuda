# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.tests import load_testsuite
import os


def load_tests(loader, tests, pattern):
    return load_testsuite(loader, os.path.dirname(__file__))
