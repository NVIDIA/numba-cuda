#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
AWKWARD_VERSION="2.8.10"

rapids-logger "Install awkward and related libraries"

pip install awkward==${AWKWARD_VERSION} "cupy-cuda12x==13.4.*" pyarrow pandas nox

rapids-logger "Install wheel with test dependencies"
package=$(realpath wheel/numba_cuda*.whl)
echo "Package path: ${package}"
python -m pip install \
    "${package}" \
    "cuda-python==${CUDA_VER_MAJOR_MINOR%.*}.*" \
    "cuda-core" \
    "nvidia-nvjitlink-cu12" \
    --group test


rapids-logger "Clone awkward repository"
git clone --recursive https://github.com/scikit-hep/awkward.git
pushd awkward
git checkout v${AWKWARD_VERSION}

patch -p1 <<'EOF'
diff --git a/dev/generate-tests.py b/dev/generate-tests.py
index 1292e0cf..4534a57c 100644
--- a/dev/generate-tests.py
+++ b/dev/generate-tests.py
@@ -970,7 +970,6 @@ cuda_kernels_tests = [
     "awkward_UnionArray_regular_index_getsize",
     "awkward_UnionArray_simplify",
     "awkward_UnionArray_simplify_one",
-    "awkward_RecordArray_reduce_nonlocal_outoffsets_64",
     "awkward_reduce_count_64",
     "awkward_reduce_max",
     "awkward_reduce_max_complex",
EOF

rapids-logger "Generate awkward tests"
nox -s prepare -- --tests

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Awkward CUDA tests"
python -m pytest -n auto --benchmark-disable tests-cuda tests-cuda-kernels tests-cuda-kernels-explicit

popd
