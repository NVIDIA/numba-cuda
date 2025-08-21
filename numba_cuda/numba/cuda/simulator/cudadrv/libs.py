# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


def check_static_lib(lib):
    raise FileNotFoundError("Linking libraries not supported by cudasim")


def get_cuda_include_dir():
    raise FileNotFoundError("CUDA includes not supported by cudasim")
