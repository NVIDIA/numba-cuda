#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

rapids-logger "Show Numba system info"
pixi run -e docs python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Build docs"
pixi run -e docs make -C docs html

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
