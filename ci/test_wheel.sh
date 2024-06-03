#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
python -m pip install \
    "numba>=0.60" \
    psutil \
    cuda-python \
    pytest

rapids-logger "Download Wheel"
RAPIDS_PY_WHEEL_NAME="numba_cuda" rapids-download-wheels-from-s3 ./dist/

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Install wheel"
python -m pip install $(echo ./dist/numba_cuda*.whl)

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
pushd numba_cuda/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-numba_cuda.xml" \
  -v
popd
