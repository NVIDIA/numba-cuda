@echo off
REM Test binaries are build taking into accoutn the CC of the GPU in the test machine
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv ^| findstr /v compute_cap ^| head -n 1 ^| sed "s/\.//"') do set GPU_CC=%%i
if "%GPU_CC%"=="" set GPU_CC=75

REM Use CC 7.0 as an alternative in fatbin testing, unless CC is 7.x
if "%GPU_CC:~0,1%"=="7" (
    set ALT_CC=80
) else (
    set ALT_CC=70
)

REM Gencode flags suitable for most tests
set GENCODE=-gencode arch=compute_%GPU_CC%,code=sm_%GPU_CC%

REM Fatbin tests need to generate code for an additional compute capability
set FATBIN_GENCODE=%GENCODE% -gencode arch=compute_%ALT_CC%,code=sm_%ALT_CC%

REM LTO-IR tests need to generate for the LTO "architecture" instead
set LTOIR_GENCODE=-gencode arch=lto_%GPU_CC%,code=lto_%GPU_CC%

REM Compile with optimization; use relocatable device code to preserve device
REM functions in the final output
set NVCC_FLAGS=-O3 -rdc true

REM Flags specific to output type
set CUBIN_FLAGS=%GENCODE% --cubin
set PTX_FLAGS=%GENCODE% -ptx
set OBJECT_FLAGS=%GENCODE% -dc
set LIBRARY_FLAGS=%GENCODE% -lib
set FATBIN_FLAGS=%FATBIN_GENCODE% --fatbin
set LTOIR_FLAGS=%LTOIR_GENCODE% -dc

set OUTPUT_DIR=.

REM Echo configuration
echo GPU CC: %GPU_CC%
echo Alternative CC: %ALT_CC%

@echo on
@REM Compile all test objects
nvcc %NVCC_FLAGS% %CUBIN_FLAGS% -o %OUTPUT_DIR%\undefined_extern.cubin undefined_extern.cu
nvcc %NVCC_FLAGS% %CUBIN_FLAGS% -o %OUTPUT_DIR%\test_device_functions.cubin test_device_functions.cu
nvcc %NVCC_FLAGS% %FATBIN_FLAGS% -o %OUTPUT_DIR%\test_device_functions.fatbin test_device_functions.cu
nvcc %NVCC_FLAGS% %PTX_FLAGS% -o %OUTPUT_DIR%\test_device_functions.ptx test_device_functions.cu
nvcc %NVCC_FLAGS% %OBJECT_FLAGS% -o %OUTPUT_DIR%\test_device_functions.o test_device_functions.cu
nvcc %NVCC_FLAGS% %LIBRARY_FLAGS% -o %OUTPUT_DIR%\test_device_functions.a test_device_functions.cu

@REM Generate LTO-IR wrapped in a fatbin
nvcc %NVCC_FLAGS% %LTOIR_FLAGS% -o %OUTPUT_DIR%\test_device_functions.ltoir.o test_device_functions.cu

@REM Generate LTO-IR in a "raw" LTO-IR container
python generate_raw_ltoir.py --arch sm_%GPU_CC% -o %OUTPUT_DIR%\test_device_functions.ltoir test_device_functions.cu