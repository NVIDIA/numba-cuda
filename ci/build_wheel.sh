#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Build wheel"
mkdir -p ./dist
python -m pip wheel . --wheel-dir=./dist -vvv --disable-pip-version-check --no-deps

package_path=$(realpath ./dist/numba_cuda-*.whl)
echo "Package path: $package_path"
echo "package_path=$package_path" >> $GITHUB_ENV
