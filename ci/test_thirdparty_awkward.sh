#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
AWKWARD_VERSION="2.8.10"

rapids-logger "Install awkward and related libraries"

pip install awkward==${AWKWARD_VERSION} cupy-cuda12x pyarrow pandas nox

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core==0.3.*" \
    "nvidia-nvjitlink-cu12" \
    --group test


rapids-logger "Clone awkward repository"
git clone --recursive https://github.com/scikit-hep/awkward.git
pushd awkward
git checkout v${AWKWARD_VERSION}

rapids-logger "Generate awkward tests"
nox -s prepare -- --tests

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Awkward CUDA tests"
python -m pytest -v -n auto tests-cuda tests-cuda-kernels tests-cuda-kernels-explicit

popd
