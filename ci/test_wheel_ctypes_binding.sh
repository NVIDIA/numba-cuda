#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

CUDA_VER_MAJOR=${CUDA_VER%.*.*}

rapids-logger "Install wheel with testing dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: $package"
python -m pip install \
    "${package}[test]" \
    cuda-python \

rapids-logger "Build tests"
PY_SCRIPT="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$PY_SCRIPT")
pushd $NUMBA_CUDA_TEST_BIN_DIR
NUMBA_CUDA_USE_NVIDIA_BINDING=0 make
popd


rapids-logger "Check GPU usage"
nvidia-smi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-logger "Show Numba system info"
NUMBA_CUDA_USE_NVIDIA_BINDING=0 python -m numba --sysinfo

rapids-logger "Run Tests"
NUMBA_CUDA_USE_NVIDIA_BINDING=0 NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR python -m pytest --pyargs numba.cuda.tests -v

popd
