#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
DEPENDENCIES=(
    "psutil"
    "pytest"
    "pytest-xdist"
    "cffi"
    "ml_dtypes"
    "python=${RAPIDS_PY_VERSION}"
    "numba-cuda"
)
rapids-mamba-retry create \
    -n test \
    --strict-channel-priority \
    --channel "`pwd`/conda-repo" \
    --channel conda-forge \
    "${DEPENDENCIES[@]}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

pip install filecheck

rapids-print-env

# The simulator doesn't actually use the test binaries, but we move into the
# test binaries folder so that we're not in the root of the repo, and therefore
# numba-cuda code from the installed package will be tested, instead of the
# code in the source repo.
rapids-logger "Move to test binaries folder"
export NUMBA_CUDA_TEST_BIN_DIR=`pwd`/testing
pushd $NUMBA_CUDA_TEST_BIN_DIR

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
export NUMBA_ENABLE_CUDASIM=1
pytest -v --pyargs numba.cuda.tests

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
