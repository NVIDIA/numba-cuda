#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

USE_PYNVJITLINK=$1

. /opt/conda/etc/profile.d/conda.sh

if [ "${CUDA_VER%.*.*}" = "11" ]; then
  CTK_PACKAGES="cudatoolkit"
else
  CTK_PACKAGES="cuda-nvcc-impl cuda-nvrtc"
fi

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    c-compiler \
    cxx-compiler \
    ${CTK_PACKAGES} \
    cuda-python \
    cuda-version=${CUDA_VER%.*} \
    make \
    psutil \
    pytest \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-mamba-retry install -c `pwd`/conda-repo numba-cuda

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

if [ "$USE_PYNVJITLINK" == true ]; then
    rapids-logger "Install pynvjitlink"
    set +u
    conda install -c rapidsai pynvjitlink
    set -u
fi

rapids-logger "Run Tests"
ENABLE_PYNVJITLINK=$USE_PYNVJITLINK python -m numba.runtests numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
