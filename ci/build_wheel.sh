#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Install build package"
python -m pip install build

rapids-logger "Build sdist and wheel"
python -m build .

wheel_path=$(realpath ./dist/numba_cuda-*.whl)
echo "Wheel path: $wheel_path"
echo "wheel_path=$wheel_path" >> $GITHUB_ENV

sdist_path=$(realpath ./dist/numba_cuda-*.tar.gz)
echo "ssdist path: $sdist_path"
echo "sdist_path=$sdist_path" >> $GITHUB_ENV
