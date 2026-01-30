#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

rapids-logger "Install wheel with test dependencies and coverage tools"
package=$(realpath "${NUMBA_CUDA_ARTIFACTS_DIR}"/*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core" \
    pytest-cov \
    coverage \
    --group test

rapids-logger "Build test binaries"
export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/testing
make -j $(nproc) -C "$NUMBA_CUDA_TEST_BIN_DIR"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests with Coverage"
python -m pytest -v --cov --pyargs numba.cuda.tests --ignore=.

rapids-logger "Generate Markdown Coverage Report"
python -m coverage report --format markdown >> $GITHUB_STEP_SUMMARY
