#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Install pixi env"
pixi install -e "${PIXI_ENV}"

rapids-logger "Show Numba system info"
pixi run -e "${PIXI_ENV}" python -m numba --sysinfo

rapids-logger "Test importing numba.cuda"
pixi run -e "${PIXI_ENV}" python -c "from numba import cuda"

rapids-logger "Run Tests"
pixi run -e "${PIXI_ENV}" simtest -n auto \
  --dist loadscope \
  --loadscope-reorder \
  --benchmark-disable -v

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
