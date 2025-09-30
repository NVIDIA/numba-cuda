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

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
export NUMBA_ENABLE_CUDASIM=1
pytest tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
