#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

rapids-logger "Install build package"
python -m pip install build

rapids-logger "Build sdist and wheel"
python -m build .
auditwheel repair -w ../final-dist dist/*.whl
echo "Repaired wheel $(ls -lh final-dist)"

wheel_path=$(realpath ./dist/numba_cuda-*.whl)
echo "Wheel path: $wheel_path"
echo "wheel_path=$wheel_path" >> $GITHUB_ENV

sdist_path=$(realpath ./dist/numba_cuda-*.tar.gz)
echo "ssdist path: $sdist_path"
echo "sdist_path=$sdist_path" >> $GITHUB_ENV
