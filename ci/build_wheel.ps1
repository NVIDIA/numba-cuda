# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function rapids-logger {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Text
    )

    # Determine padding and box width
    $padding = 2
    $boxWidth = $Text.Length + ($padding * 2)
    $topBottom = '+' + ('-' * $boxWidth) + '+'
    $middle = '|' + (' ' * $padding) + $Text + (' ' * $padding) + '|'

    # Print the box in green
    Write-Host $topBottom -ForegroundColor Green
    Write-Host $middle    -ForegroundColor Green
    Write-Host $topBottom -ForegroundColor Green
}

rapids-logger "Install build package"
python -m pip install build

rapids-logger "Build sdist and wheel"
python -m build .

$wheel_path = Resolve-Path dist\numba_cuda*.whl | Select-Object -ExpandProperty Path
echo "Wheel path: $wheel_path"
echo "wheel_path=$wheel_path" >> $GITHUB_ENV
