#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install docs dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n docs \
    make \
    psutil \
    sphinx \
    sphinx_rtd_theme \
    numpydoc \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

# Detect system architecture to set conda repo path
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_SUFFIX="amd64"
elif [[ "$ARCH" == "aarch64" ]]; then
    ARCH_SUFFIX="arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

rapids-mamba-retry install -c `pwd`/conda-repo-py${RAPIDS_PY_VERSION}-${ARCH_SUFFIX} numba-cuda

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
