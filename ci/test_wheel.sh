#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
CUDA_VER_MAJOR=${CUDA_VER%.*.*}


rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"

DEPENDENCIES=(
    "${package}[cu${CUDA_VER_MAJOR},test-cu${CUDA_VER_MAJOR}]"
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*"
    "cuda-core==0.3.*"
)

# Constrain oldest supported dependencies for testing
if [ "${RAPIDS_DEPENDENCIES:-}" = "oldest" ]; then
    DEPENDENCIES+=("numba==0.60.0")
else
    DEPENDENCIES+=("numba<0.62.0")
fi

python -m pip install "${DEPENDENCIES[@]}"

rapids-logger "Test importing numba.cuda"
python -c "from numba import cuda"

GET_TEST_BINARY_DIR="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

rapids-logger "Build tests"
export NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$GET_TEST_BINARY_DIR")
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

rapids-logger "Run Tests"
python -m pytest --pyargs numba.cuda.tests -v

popd
