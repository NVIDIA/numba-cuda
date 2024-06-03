#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Build wheel"
mkdir -p ./dist
python -m pip wheel . --wheel-dir=./dist -vvv --disable-pip-version-check --no-deps

python -m auditwheel repair -w ./final_dist ./dist/*

rapids-logger "Upload Wheel"
RAPIDS_PY_WHEEL_NAME="numba_cuda" rapids-upload-wheels-to-s3 ./final_dist
