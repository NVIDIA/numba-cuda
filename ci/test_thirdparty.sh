#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}[test]" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core==0.3.*" \
    "cudf-cu12"


rapids-logger "Shallow clone cuDF repository"
git clone --depth 1 git@github.com:rapidsai/cudf.git

pushd cudf

rapids-logger "Check GPU usage"
nvidia-smi


rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Scalar UDF tests"
py.test python/cudf/cudf/tests/test_udf_masked_ops.py

rapids-logger "Run GroupBy UDF tests"
py.test python/cudf/cudf/tests/test_groupby.py -k test_groupby_apply_jit

popd
