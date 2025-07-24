#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

if [ "${CUDA_VER%.*.*}" = "11" ]; then
  CTK_PACKAGES="cudatoolkit=11"
else
  CTK_PACKAGES="cuda-cccl cuda-nvcc-impl cuda-nvrtc libcurand-dev cuda-cuobjdump"
  apt-get update
  apt remove --purge `dpkg --get-selections | grep cuda-nvvm | awk '{print $1}'` -y
  apt remove --purge `dpkg --get-selections | grep cuda-nvrtc | awk '{print $1}'` -y
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
    pytest-xdist \
    cffi \
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


GET_TEST_BINARY_DIR="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
print(test_dir)
"

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
if [ "${CUDA_VER_MAJOR_MINOR%.*}" == "11" ]
then
  rapids-logger "Skipping test build for CUDA 11"
else
  rapids-logger "Build tests"

  export NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$GET_TEST_BINARY_DIR")
  pushd $NUMBA_CUDA_TEST_BIN_DIR
  make
  popd
fi


rapids-logger "Run Tests"
pytest --pyargs numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
