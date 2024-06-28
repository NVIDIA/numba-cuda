cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  numba_cuda
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

install(FILES site-packages/_numba_cuda_redirector.py
              site-packages/_numba_cuda_redirector.pth
              DESTINATION ${SKBUILD_PURELIB_DIR})
