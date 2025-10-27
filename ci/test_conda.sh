#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

DISTRO=`cat /etc/os-release | grep "^ID=" | awk 'BEGIN {FS="="} { print $2 }'`

if [ "$DISTRO" = "ubuntu" ]; then
  apt-get update
  apt remove --purge `dpkg --get-selections | grep cuda-nvvm | awk '{print $1}'` -y
  apt remove --purge `dpkg --get-selections | grep cuda-nvrtc | awk '{print $1}'` -y
fi

# Constrain oldest supported dependencies for testing
if [ "${RAPIDS_DEPENDENCIES:-}" = "oldest" ]; then
    # add to the default environment's dependencies
    pixi add "numba=0.60.0"
fi

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Show Numba system info"
pixi run -e "${PIXI_ENV}" python -m numba --sysinfo

rapids-logger "Test importing numba.cuda"
pixi run -e "${PIXI_ENV}" python -c "from numba import cuda"

rapids-logger "Run Tests"
pixi run -e "${PIXI_ENV}" test -n auto \
  --dist loadscope \
  --loadscope-reorder \
  -v

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
