# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda import config
from collections import namedtuple

_nrt_mstats = namedtuple("nrt_mstats", ["alloc", "free", "mi_alloc", "mi_free"])


class RTSys:
    def __init__(self, *args, **kwargs):
        pass

    def memsys_enable_stats(self):
        pass

    def memsys_disable_stats(self):
        pass

    def get_allocation_stats(self):
        return _nrt_mstats(alloc=0, free=0, mi_alloc=0, mi_free=0)


rtsys = RTSys()

config.CUDA_NRT_STATS = False
config.CUDA_ENABLE_NRT = False
