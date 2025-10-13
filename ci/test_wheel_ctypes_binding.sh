#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR=${CUDA_VER%.*.*}

rapids-logger "Install wheel with testing dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: $package"
python -m pip install \
    "${package}" \
    cuda-python \
    --group test

# FIXME: Find a way to build the tests that does not depend on the CUDA Python bindings
#rapids-logger "Build tests"
rapids-logger "Copy and cd into test binaries dir"
export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/testing
pushd $NUMBA_CUDA_TEST_BIN_DIR
# make -j $(nproc)

# Prevent the testsuite trying to use the test binaries
unset NUMBA_CUDA_TEST_BIN_DIR

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
NUMBA_CUDA_USE_NVIDIA_BINDING=0 python -m numba --sysinfo

rapids-logger "Run Tests"
# NUMBA_CUDA_USE_NVIDIA_BINDING=0 NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR python -m pytest --pyargs numba.cuda.tests -v
NUMBA_CUDA_USE_NVIDIA_BINDING=0 python -m pytest -v

popd
