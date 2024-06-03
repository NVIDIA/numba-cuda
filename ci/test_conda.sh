#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    c-compiler \
    cxx-compiler \
    cuda-nvcc \
    cuda-nvrtc \
    cuda-python \
    cuda-version=${RAPIDS_CUDA_VERSION%.*} \
    "numba>=0.60" \
    make \
    psutil \
    pytest \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  numba_cuda

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
pushd numba_cuda/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-numba_cuda.xml" \
  -v
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
