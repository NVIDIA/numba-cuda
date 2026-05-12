#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

NVMATH_PYTHON_VERSION="0.9.0"
# The commit on Github corresponding to 0.9.0
NVMATH_PYTHON_SHA="48f5b643c8b7c04f8be1745b9b700eae17af7319"

rapids-logger "Install nvmath-python"

pip install nvmath-python[cu12-dx]==${NVMATH_PYTHON_VERSION}

rapids-logger "Remove Extraneous numba-cuda"
pip uninstall -y numba-cuda

rapids-logger "Install wheel with test dependencies"
package=$(realpath "${NUMBA_CUDA_ARTIFACTS_DIR}"/*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "nvidia-nvjitlink-cu12" \
    --group test

rapids-logger "Verify environment consistency"
pip check

rapids-logger "Shallow clone nvmath-python repository"
git clone https://github.com/NVIDIA/nvmath-python.git
pushd nvmath-python
git checkout ${NVMATH_PYTHON_SHA}

rapids-logger "Install nvmath-python test dependencies"
pip install cffi hypothesis opt_einsum packaging psutil pytest pytest-repeat pytest-xdist scipy
pip install "nvidia-mathdx~=25.6.0"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run nvmath-python device tests"
pushd tests
# Required for nvmath-python to locate pip-installed MathDx
export MATHDX_HOME=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/nvidia/mathdx
python -m pytest -n auto -k "not (perf or benchmark)" nvmath_tests/device --tb=native -x

popd
popd
