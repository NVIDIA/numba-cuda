# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda import config


class RTSys:
    def __init__(self, *args, **kwargs):
        pass

    def memsys_enable_stats(self):
        pass

    def get_allocation_stats(self):
        pass


rtsys = RTSys()

config.CUDA_NRT_STATS = False
config.CUDA_ENABLE_NRT = False
