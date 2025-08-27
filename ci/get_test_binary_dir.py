# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda
import os

root = os.path.dirname(numba_cuda.__file__)
test_dir = os.path.join(
    root, "numba", "cuda", "tests", "test_binary_generation"
)
print(test_dir)
