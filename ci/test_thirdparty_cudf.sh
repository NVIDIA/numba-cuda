#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}

rapids-logger "Install cuDF Wheel"

pip install "cudf-cu12==25.12.*"


rapids-logger "Remove Extraneous numba-cuda"
pip uninstall -y numba-cuda

rapids-logger "Install wheel with test dependencies"
package=$(realpath "${NUMBA_CUDA_ARTIFACTS_DIR}"/*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core" \
    "nvidia-nvjitlink-cu12" \
    --group test



rapids-logger "Shallow clone cuDF repository"
git clone --single-branch --branch 'release/25.12' https://github.com/rapidsai/cudf.git

# TODO: remove the patch and its application after 26.02 is released
patchfile="${PWD}/ci/patches/cudf_numba_cuda_compatibility.patch"
pushd "$(python -c 'import site; print(site.getsitepackages()[0])')"
# strip 3 slahes to apply from the root of the install
patch -p3 < "${patchfile}"
popd

pushd cudf

rapids-logger "Check GPU usage"
nvidia-smi


rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Scalar UDF tests"
python -m pytest python/cudf/cudf/tests/dataframe/methods/test_apply.py -W ignore::UserWarning

rapids-logger "Run GroupBy UDF tests"
# Run JIT engine tests and tests that check jittability before falling back
python -m pytest python/cudf/cudf/tests/groupby/test_apply.py -k test_groupby_apply -W ignore::UserWarning

rapids-logger "Run NRT Stats Counting tests"
python -m pytest python/cudf/cudf/tests/private_objects/test_nrt_stats.py  -W ignore::UserWarning


popd
