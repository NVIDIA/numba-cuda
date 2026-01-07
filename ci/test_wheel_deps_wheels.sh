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
python -m pip install "${package}[cu${CUDA_VER_MAJOR},test-cu${CUDA_VER_MAJOR}]" "numba=<0.63.0"

rapids-logger "Build tests"
PY_SCRIPT="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"
NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$PY_SCRIPT")
pushd $NUMBA_CUDA_TEST_BIN_DIR
make
popd


rapids-logger "Check GPU usage"
nvidia-smi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-logger "Show Numba system info"
python -m numba --sysinfo

apt-get update
nvvm_pkgs=$(dpkg --get-selections | awk '/nvvm/ {print $1}')
nvrtc_pkgs=$(dpkg --get-selections | awk '/nvrtc/ {print $1}')

if [[ -z "$nvvm_pkgs" || -z "$nvrtc_pkgs" ]]; then
    echo "Expected both nvvm and nvrtc packages to be present, but at least one was missing"
    exit 1
fi

apt remove --purge -y $nvvm_pkgs
apt remove --purge -y $nvrtc_pkgs

rapids-logger "Run Tests"
NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR python -m pytest --pyargs numba.cuda.tests -v

popd
