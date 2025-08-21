# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import config

rtsys = None

config.CUDA_NRT_STATS = False
config.CUDA_ENABLE_NRT = False
