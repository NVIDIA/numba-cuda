#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry build conda/recipes/numba-cuda

package_path=(/tmp/conda-bld-output/noarch/numba-cuda-*.tar.bz2)
echo "package_path=$package_path" >> $GITHUB_ENV
