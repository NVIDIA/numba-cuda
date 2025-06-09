#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

if [ "${CUDA_VER%.*.*}" = "11" ]; then
  CTK_PACKAGES="cudatoolkit"
else
  CTK_PACKAGES="cuda-nvcc-impl cuda-nvrtc cuda-cuobjdump libcurand-dev"
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
    cffi \
    python=${RAPIDS_PY_VERSION} \
    numpy=2.2

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


rapids-logger "Install pynvjitlink"
set +u
rapids-mamba-retry install -c rapidsai pynvjitlink
set -u

rapids-logger "Build tests"

PY_SCRIPT="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$PY_SCRIPT")
pushd $NUMBA_CUDA_TEST_BIN_DIR
make
popd


rapids-logger "Run Tests"
NUMBA_CUDA_ENABLE_PYNVJITLINK=1 NUMBA_CUDA_TEST_BIN_DIR=$NUMBA_CUDA_TEST_BIN_DIR python -m numba.runtests numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
