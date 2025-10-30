# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import importlib.resources

__version__ = (
    importlib.resources.files("numba_cuda")
    .joinpath("VERSION")
    .read_text()
    .strip()
)
