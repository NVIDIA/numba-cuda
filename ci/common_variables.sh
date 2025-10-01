#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail
set -x
export NUMBA_CUDA_TEST_DIR=$PWD/tests
export NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_DIR/test_binary_generation
rapids-logger "NUMBA_CUDA_TEST_DIR: $NUMBA_CUDA_TEST_DIR"
rapids-logger "NUMBA_CUDA_TEST_BIN_DIR: $NUMBA_CUDA_TEST_BIN_DIR"
find ./tests -type f
