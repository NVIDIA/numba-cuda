# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
from pathlib import Path

import pytest


_test_bin_dir = Path(__file__).resolve().parent / "testing"
if (_test_bin_dir / "test_device_functions.cu").exists():
    os.environ.setdefault("NUMBA_CUDA_TEST_BIN_DIR", str(_test_bin_dir))


def pytest_addoption(parser):
    parser.addoption(
        "--dump-failed-filechecks",
        action="store_true",
        help="Dump reproducers for FileCheck tests that fail.",
    )


@pytest.fixture(scope="class")
def initialize_from_pytest_config(request):
    """
    Fixture to initialize the test case with pytest configuration options.
    """
    request.cls._dump_failed_filechecks = request.config.getoption(
        "dump_failed_filechecks"
    )
