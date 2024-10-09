#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
python -m pip install \
    psutil \
    cuda-python \
    pytest

rapids-logger "Install pynvjitlink"
python -m pip install pynvjitlink-cu12

rapids-logger "Build tests"
PY_SCRIPT="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

TEST_DIR=$(python -c "$PY_SCRIPT")
pushd $TEST_DIR
make
popd

rapids-logger "Install wheel"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: $package"
python -m pip install $package

rapids-logger "Check GPU usage"
nvidia-smi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
ENABLE_PYNVJITLINK=1 python -m numba.runtests numba.cuda.tests -v

popd
