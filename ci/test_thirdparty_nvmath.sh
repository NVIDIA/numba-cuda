#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

NVMATH_PYTHON_VERSION="0.6.0"
# The commit on Github corresponding to 0.6.0
NVMATH_PYTHON_SHA="6bddfa71c39c07804127adeb23f5b0d2168ae38c"

rapids-logger "Install nvmath-python"

pip install nvmath-python[cu12,dx]==${NVMATH_PYTHON_VERSION}

rapids-logger "Remove Extraneous numba-cuda"
pip uninstall -y numba-cuda

rapids-logger "Install wheel with test dependencies"
package=$(realpath "${NUMBA_CUDA_ARTIFACTS_DIR}"/*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core" \
    "nvidia-nvjitlink-cu12" \
    --group test


rapids-logger "Shallow clone nvmath-python repository"
git clone https://github.com/NVIDIA/nvmath-python.git
pushd nvmath-python
git checkout ${NVMATH_PYTHON_SHA}

rapids-logger "Install nvmath-python test dependencies"
pip install -r requirements/pip/tests.txt
pip install nvidia-mathdx
pip install nvidia-cutlass

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run nvmath-python device tests"
pushd tests
# Required for nvmath-python to locate pip-install MathDx
export SYS_PREFIX=`python -c "import sys; print(sys.prefix)"`
export MATHDX_HOME=${SYS_PREFIX}/lib/python3.13/site-packages/nvidia/mathdx
python -m pytest nvmath_tests/device --tb=native -x

popd
popd
