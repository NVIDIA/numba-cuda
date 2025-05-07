#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
python -m pip install \
    psutil \
    cffi \
    "cuda-python==${CUDA_VER_MAJOR_MINOR}.*" \
    pytest


GET_TEST_BINARY_DIR="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

if [ "${CUDA_VER_MAJOR_MINOR%.*}" == "11" ]
then
  rapids-logger "Skipping test build for CUDA 11"
else
  rapids-logger "Build tests"

  export NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$GET_TEST_BINARY_DIR")
  pushd $NUMBA_CUDA_TEST_BIN_DIR
  make
  popd
fi


rapids-logger "Install wheel"
package=$(realpath wheel/numba_cuda*.whl)
echo "Wheel path: $package"
python -m pip install $package

rapids-logger "Check GPU usage"
nvidia-smi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
python -m numba.runtests numba.cuda.tests -v

popd
