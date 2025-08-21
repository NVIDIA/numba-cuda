# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


class CudaSupportError(RuntimeError):
    pass


class NvrtcError(Exception):
    pass
