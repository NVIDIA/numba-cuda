#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    c-compiler \
    cxx-compiler \
    cuda-nvcc-impl \
    cuda-nvrtc \
    cuda-python \
    cuda-version=${RAPIDS_CUDA_VERSION%.*} \
    make \
    psutil \
    pytest \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

package=$(realpath conda-package/numba-cuda-*.tar.bz2)
echo "Package path: $package"
rapids-mamba-retry install $package

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

rapids-logger "Run Tests"
python -m numba.runtests numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
