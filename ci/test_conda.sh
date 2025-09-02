#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail
set -x

. /opt/conda/etc/profile.d/conda.sh

CTK_PACKAGE_DEPENDENCIES=(
    "cuda-cccl"
    "cuda-nvcc-impl"
    "cuda-nvrtc"
    "libcurand-dev"
    "cuda-cuobjdump"
)

DISTRO=`cat /etc/os-release | grep "^ID=" | awk 'BEGIN {FS="="} { print $2 }'`

if [ "$DISTRO" = "ubuntu" ]; then
  apt-get update
  apt remove --purge `dpkg --get-selections | grep cuda-nvvm | awk '{print $1}'` -y
  apt remove --purge `dpkg --get-selections | grep cuda-nvrtc | awk '{print $1}'` -y
fi

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
DEPENDENCIES=(
    "c-compiler"
    "cxx-compiler"
    "${CTK_PACKAGE_DEPENDENCIES[@]}"
    "cuda-python"
    "cuda-version=${CUDA_VER%.*}"
    "make"
    "numba-cuda"
    "psutil"
    "pytest"
    "pytest-xdist"
    "cffi"
    "ml_dtypes"
    "python=${RAPIDS_PY_VERSION}"
)
# Constrain oldest supported dependencies for testing
if [ "${RAPIDS_DEPENDENCIES:-}" = "oldest" ]; then
    DEPENDENCIES+=("numba==0.60.0")
fi

rapids-mamba-retry create \
    -n test \
    --strict-channel-priority \
    --channel "`pwd`/conda-repo" \
    --channel conda-forge \
    "${DEPENDENCIES[@]}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

pip install filecheck

# Detect system architecture to set conda repo path
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_SUFFIX="amd64"
elif [[ "$ARCH" == "aarch64" ]]; then
    ARCH_SUFFIX="arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

rapids-mamba-retry install -c `pwd`/conda-repo-py${RAPIDS_PY_VERSION}-${ARCH_SUFFIX} numba-cuda

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

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


rapids-logger "Run Tests"
pytest --pyargs numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
