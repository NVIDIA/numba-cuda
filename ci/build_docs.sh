#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

source ci/common_variables.sh

rapids-logger "Install docs dependencies"
# TODO: Replace with rapids-dependency-file-generator
DEPENDENCIES=(
    "make"
    "psutil"
    "sphinx"
    "sphinx_rtd_theme"
    "numpydoc"
    "python=${RAPIDS_PY_VERSION}"
    "numba-cuda"
)
rapids-mamba-retry create \
    -n docs \
    --strict-channel-priority \
    --channel "`pwd`/conda-repo" \
    --channel conda-forge \
    "${DEPENDENCIES[@]}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

pip install nvidia-sphinx-theme

rapids-print-env

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Build docs"
pushd docs
make html

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
