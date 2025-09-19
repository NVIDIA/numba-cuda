#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR=${CUDA_VER%.*.*}

rapids-logger "Install wheel with testing dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: $package"
python -m pip install \
    "${package}[test]" \
    cuda-python \

# FIXME: Find a way to build the tests that does not depend on the CUDA Python bindings
#rapids-logger "Build tests"
#PY_SCRIPT="
#import numba_cuda
#root = numba_cuda.__file__.rstrip('__init__.py')
#test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
#print(test_dir)
#"
#
#NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$PY_SCRIPT")
#pushd $NUMBA_CUDA_TEST_BIN_DIR
## ctypes mode removed
#popd


rapids-logger "Check GPU usage"
nvidia-smi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
# NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR python -m pytest --pyargs numba.cuda.tests -v
python -m pytest --pyargs numba.cuda.tests -v

popd
