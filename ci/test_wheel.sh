#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"

DEPENDENCIES=(
    "${package}"
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*"
    "cuda-core>=0.3.0,<1.0.0"
    "--group"
    "test"
)

# Constrain oldest supported dependencies for testing
if [ "${RAPIDS_DEPENDENCIES:-}" = "oldest" ]; then
    DEPENDENCIES+=("numba==0.60.0")
fi

python -m pip install "${DEPENDENCIES[@]}"

rapids-logger "Build tests"
export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/testing
pushd $NUMBA_CUDA_TEST_BIN_DIR
make -j $(nproc)

rapids-logger "Test importing numba.cuda"
python -c "from numba import cuda"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
python -m pytest -v

popd
