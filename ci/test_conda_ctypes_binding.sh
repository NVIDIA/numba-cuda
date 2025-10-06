#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

CTK_PACKAGE_DEPENDENCIES=(
    "cuda-nvcc-impl"
    "cuda-nvrtc"
    "cuda-cuobjdump"
    "libcurand-dev"
)

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
DEPENDENCIES=(
    "c-compiler"
    "cxx-compiler"
    "${CTK_PACKAGE_DEPENDENCIES[@]}"
    "cuda-python"
    "cuda-version=${CUDA_VER%.*}"
    "make"
    "psutil"
    "pytest"
    "pytest-xdist"
    "cffi"
    "ml_dtypes"
    "python=${RAPIDS_PY_VERSION}"
    "numba-cuda"
)
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

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Build test binaries"

export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/test_binary_generation
pushd $NUMBA_CUDA_TEST_BIN_DIR
make

rapids-logger "Run Tests"
NUMBA_CUDA_USE_NVIDIA_BINDING=0 NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR pytest --pyargs numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
