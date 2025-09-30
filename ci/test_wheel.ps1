# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Copyright (c) 2023-2024, NVIDIA CORPORATION

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


$CUDA_VER_MAJOR_MINOR = ($env:CUDA_VER -split '\.')[0..1] -join '.'
$CUDA_VER_MAJOR = ($env:CUDA_VER -split '\.')[0] -join '.'

rapids-logger "Install wheel with test dependencies"
$package = Resolve-Path wheel\numba_cuda*.whl | Select-Object -ExpandProperty Path
echo "Package path: $package"
python -m pip install "${package}[cu${CUDA_VER_MAJOR},test-cu${CUDA_VER_MAJOR}]"

# Fix for llvmlite 0.45.0 Windows wheel missing dependencies
# See: https://github.com/numba/llvmlite/issues/1297
rapids-logger "Install fixed llvmlite 0.45.0 Windows wheel"

# Check if the fixed wheel was downloaded by the workflow
$fixed_wheel_path = "llvmlite-fixed.whl"

if (Test-Path $fixed_wheel_path) {
    rapids-logger "Installing fixed llvmlite wheel"
    python -m pip install --force-reinstall --no-deps $fixed_wheel_path
    rapids-logger "Successfully installed fixed llvmlite wheel"

    # Clean up the wheel file
    Remove-Item $fixed_wheel_path -Force -ErrorAction SilentlyContinue
} else {
    rapids-logger "Fixed wheel not found, falling back to version pinning"
    python -m pip install "llvmlite<0.45" "numba==0.61.*"
}


rapids-logger "Build tests"
$NUMBA_CUDA_TEST_BIN_DIR = (python ci\get_test_binary_dir.py)
echo "Test binary dir: $NUMBA_CUDA_TEST_BIN_DIR"
pushd $NUMBA_CUDA_TEST_BIN_DIR
Get-Location

cmd.exe /c '.\build.bat'
popd

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
python -m pytest --pyargs numba.cuda.tests -v
