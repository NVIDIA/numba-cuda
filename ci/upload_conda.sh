#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PKGS_TO_UPLOAD=$(rapids-find-anaconda-uploads.py $RAPIDS_CONDA_BLD_OUTPUT_DIR)

rapids-retry anaconda \
    -t $CONDA_TOKEN \
    upload \
    --skip-existing \
    --no-progress \
    ${PKGS_TO_UPLOAD}
