cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  numba_cuda
  VERSION 0.1.5
  LANGUAGES C
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_extras MODULE numba_cuda/numba/cuda/cudadrv/_extras.c WITH_SOABI)

target_compile_options(_extras PRIVATE -Werror -Wall)

install(TARGETS _extras LIBRARY DESTINATION numba_cuda/numba/cuda/cudadrv)
install(FILES site-packages/_numba_cuda_redirector.py
              site-packages/_numba_cuda_redirector.pth
              DESTINATION ${SKBUILD_PLATLIB_DIR})
