#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

NVMATH_PYTHON_VERSION="0.6.*"

rapids-logger "Install nvmath-python"

pip install nvmath-python[cu12,dx]==${NVMATH_PYTHON_VERSION}

rapids-logger "Remove Extraneous numba-cuda"
pip uninstall -y numba-cuda

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core==0.3.*" \
    "nvidia-nvjitlink-cu12" \
    --group test


rapids-logger "Shallow clone nvmath-python repository"
git clone --single-branch --branch 'release-0.6.x' https://github.com/NVIDIA/nvmath-python.git

rapids-logger "Install nvmath-python test dependencies"

pip install nvmath-python/requirements/pip/tests.txt

pushd nvmath-python/tests

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run nvmath-python device tests"
# Required for nvmath-python to locate pip-install MathDx
export MATHDX_HOME=${CONDA_PREFIX}/lib/python3.13/site-packages/nvidia/mathdx
python -m pytest nvmath_tests/device

popd
