# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


class CUDADispatcher:
    """
    Dummy class so that consumers that try to import the real CUDADispatcher
    do not get an import failure when running with the simulator.
    """

    ...
