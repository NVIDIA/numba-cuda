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
$CUDA_VER_MAJOR = ($CUDA_VER -split '\.')[0] -join '.'

rapids-logger "Install wheel with test dependencies"
$package = Resolve-Path wheel\numba_cuda*.whl | Select-Object -ExpandProperty Path
echo "Package path: $package"
python -m pip install "$package[test]" "cuda-python==$CUDA_VER_MAJOR" "cuda-core==0.3.*"

# GET_TEST_BINARY_DIR="
# import numba_cuda
# root = numba_cuda.__file__.rstrip('__init__.py')
# test_dir = root + \"numba/cuda/tests/test_binary_generation/\"
# print(test_dir)
# "

# rapids-logger "Build tests"
# export NUMBA_CUDA_TEST_BIN_DIR=$(python -c "$GET_TEST_BINARY_DIR")
# pushd $NUMBA_CUDA_TEST_BIN_DIR
# make
# popd


rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
python -m pytest --pyargs numba.cuda.tests -v
