#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

CUDA_VER_MAJOR_MINOR=${CUDA_VER%.*}
AWKWARD_VERSION="2.8.10"

rapids-logger "Install awkward and related libraries"

pip install awkward==${AWKWARD_VERSION} cupy-cuda12x pyarrow pandas nox

rapids-logger "Install wheel with test dependencies"
package=$(realpath "${NUMBA_CUDA_ARTIFACTS_DIR}"/*.whl)
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

# Avoid tests that are unreliable on the CUDA target:
#
# - awkward_RecordArray_reduce_nonlocal_outoffsets_64
# - test_3459_virtualarray_with_cuda
#
# as per discussion in https://github.com/scikit-hep/awkward/discussions/3587


rapids-logger "Patch awkward tests"

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
diff --git a/tests-cuda/test_3459_virtualarray_with_cuda.py b/tests-cuda/test_3459_virtualarray_with_cuda.py
index e2bcab12..c82f63c3 100644
--- a/tests-cuda/test_3459_virtualarray_with_cuda.py
+++ b/tests-cuda/test_3459_virtualarray_with_cuda.py
@@ -9,6 +9,7 @@ import awkward as ak
 from awkward._nplikes.cupy import Cupy
 from awkward._nplikes.virtual import VirtualNDArray

+pytestmark = pytest.mark.skip("temporarily skipping all tests in this module")

 @pytest.fixture(scope="function", autouse=True)
 def cleanup_cuda():
diff --git a/tests-cuda/test_3149_complex_reducers.py b/tests-cuda/test_3149_complex_reducers.py
index 39080a34..0eb3940f 100644
--- a/tests-cuda/test_3149_complex_reducers.py
+++ b/tests-cuda/test_3149_complex_reducers.py
@@ -544,6 +544,7 @@ def test_block_boundary_prod_complex12():
     del cuda_content, cuda_depth1


+@pytest.mark.skip("Intermittent failures")
 def test_block_boundary_prod_complex13():
     rng = np.random.default_rng(seed=42)
     array = rng.integers(50, size=1000)
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
