#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

PY_SCRIPT="
import numba_cuda
root = numba_cuda.__file__.rstrip('__init__.py')
test_dir = root + numba/cuda/tests/test_binary_generation/
print(test_dir)
"

TEST_DIR=$(python -c "$PY_SCRIPT")
pushd $TEST_DIR
make
popd
