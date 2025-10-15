#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
CUDA_VER_MAJOR=${CUDA_VER%.*.*}

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"
# TODO: control minor version pinning to honor TEST_MATRIX once the cuda-toolkit metapackage is up
python -m pip install "${package}[cu${CUDA_VER_MAJOR}]" --group "test-cu${CUDA_VER_MAJOR}"

rapids-logger "Build test binaries"

export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/testing
pushd $NUMBA_CUDA_TEST_BIN_DIR
make -j $(nproc)

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

# remove cuda-nvvm-12-5 leaving libnvvm.so from nvidia-cuda-nvcc-cu12 only
apt-get update
apt remove --purge `dpkg --get-selections | grep cuda-nvvm | awk '{print $1}'` -y
apt remove --purge `dpkg --get-selections | grep cuda-nvrtc | awk '{print $1}'` -y

rapids-logger "Run Tests"
NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR pytest -v --pyargs numba.cuda.tests

popd
