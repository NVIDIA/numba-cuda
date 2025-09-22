# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda
import os
import sys

root = os.path.dirname(numba_cuda.__file__)
print("root:" + root)
test_dir = os.path.join(
    root, "numba", "cuda", "tests", "test_binary_generation"
)

if os.path.isdir(test_dir):
    print("test_dir exists:" + test_dir)
else:
    sys.exit(f"An error occured. {test_dir} does NOT exist.")
