# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda
import os
import sys

root = os.path.dirname(numba_cuda.__file__)
test_dir = os.path.join(
    root, "numba", "cuda", "tests", "test_binary_generation"
)

if os.path.isdir(test_dir):
    print(test_dir)
else:
    sys.exit(f"ERROR: '{test_dir}' does NOT exist.")
