#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

## Usage
# bash update-version.sh <new_version>


# Format is MACRO.MESO.MICRO - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > numba_cuda/VERSION
