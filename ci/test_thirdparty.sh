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


rapids-logger "Install cuDF Wheel"
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*"

rapids-logger "Shallow clone cuDF repository"
git clone --single-branch --branch 'branch-25.06' https://github.com/rapidsai/cudf.git


pushd cudf
git checkout branch-25.06

rapids-logger "Check GPU usage"
nvidia-smi


rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Scalar UDF tests"
py.test python/cudf/cudf/tests/test_udf_masked_ops.py

rapids-logger "Run GroupBy UDF tests"
py.test python/cudf/cudf/tests/test_groupby.py -k test_groupby_apply_jit

popd
