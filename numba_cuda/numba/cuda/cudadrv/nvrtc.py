# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.cudadrv.error import (
    CCSupportError,
    NvrtcError,
)
from numba.cuda import config
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.utils import _readenv

import os
import warnings
import functools

NVRTC_EXTRA_SEARCH_PATHS = _readenv(
    "NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS", str, ""
) or getattr(config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS", "")
if not hasattr(config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS"):
    config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = NVRTC_EXTRA_SEARCH_PATHS

try:
    # Prefer cuda.core for compilation
    from cuda.core.experimental import Program, ProgramOptions  # type: ignore

    _HAVE_CORE = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_CORE = False

try:
    # Use bindings for version / supported archs and as fallback for gaps
    from cuda.bindings import nvrtc as bindings_nvrtc  # type: ignore

    _HAVE_BINDINGS = True
except Exception:  # pragma: no cover - environment dependent
    bindings_nvrtc = None
    _HAVE_BINDINGS = False


@functools.cache
def _get_nvrtc_version():
    retcode, major, minor = bindings_nvrtc.nvrtcVersion()
    if retcode != bindings_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"{retcode.name} when calling nvrtcVersion()")
    return (major, minor)


def compile(src, name, cc, ltoir=False):
    """
    Compile a CUDA C/C++ source to PTX or LTOIR for a given compute capability.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param cc: A tuple ``(major, minor)`` of the compute capability
    :type cc: tuple
    :param ltoir: Compile into LTOIR if True, otherwise into PTX
    :type ltoir: bool
    :return: The compiled PTX and compilation log
    :rtype: tuple
    """
    version = _get_nvrtc_version()

    ver_str = lambda version: ".".join(str(v) for v in version)
    supported_ccs = get_supported_ccs()
    try:
        found = max(filter(lambda v: v <= cc, [v for v in supported_ccs]))
    except ValueError:
        raise RuntimeError(
            f"Device compute capability {ver_str(cc)} is less than the "
            f"minimum supported by NVRTC {ver_str(version)}. Supported "
            "compute capabilities are "
            f"{', '.join([ver_str(v) for v in supported_ccs])}."
        )

    if found != cc:
        warnings.warn(
            f"Device compute capability {ver_str(cc)} is not supported by "
            f"NVRTC {ver_str(version)}. Using {ver_str(found)} instead."
        )

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (rdc) is needed to prevent device functions
    #   being optimized away.
    major, minor = found

    arch = f"sm_{major}{minor}"

    cuda_include_dir = get_cuda_paths()["include_dir"].info
    cuda_includes = [f"{cuda_include_dir}"]

    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
    numba_cuda_path = os.path.dirname(cudadrv_path)

    nvrtc_ver_major = version[0]
    if nvrtc_ver_major == 12:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '12')}"
        # For CUDA 12 wheels, `cuda_include_dir` is `site-packages/nvidia/cuda_runtime/include`
        # We need to find CCCL at `site-packages/nvidia/cuda_cccl/include`
        # For CUDA 12 conda / system install, CCCL is just in the `include` directory
        cuda_includes.append(
            f"{os.path.join(cuda_include_dir, '..', '..', 'cuda_cccl', 'include')}"
        )
    elif nvrtc_ver_major == 13:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '13')}"
        # For CUDA 13 wheels, `cuda_include_dir` is `site-packages/nvidia/cu13/include`
        # We need to find CCCL at `site-packages/nvidia/cu13/include/cccl`
        # For CUDA 13 conda / system install, CCCL is in the `include/cccl` directory
        cuda_includes.append(f"{os.path.join(cuda_include_dir, 'cccl')}")

    if config.CUDA_NVRTC_EXTRA_SEARCH_PATHS:
        extra_includes = config.CUDA_NVRTC_EXTRA_SEARCH_PATHS.split(":")
    else:
        extra_includes = []

    nrt_include = os.path.join(numba_cuda_path, "memory_management")

    includes = [numba_include, *cuda_includes, nrt_include, *extra_includes]


    options = ProgramOptions(
        arch=arch,
        include_path=includes,
        relocatable_device_code=True,
        link_time_optimization=ltoir,
        name=name,
    )

    class Logger:
        def __init__(self):
            self.log = []

        def write(self, msg):
            self.log.append(msg)

    logger = Logger()
    if isinstance(src, bytes):
        src = src.decode("utf8")

    prog = Program(src, "c++", options=options)
    result = prog.compile("ltoir" if ltoir else "ptx", logs=logger)
    log = ""
    if logger.log:
        log = logger.log
        joined_logs = "\n".join(log)
        warnings.warn(f"NVRTC log messages: {joined_logs}")
    return result, log


def find_closest_arch(mycc):
    """
    Given a compute capability, return the closest compute capability supported
    by the CUDA toolkit.

    :param mycc: Compute capability as a tuple ``(MAJOR, MINOR)``
    :return: Closest supported CC as a tuple ``(MAJOR, MINOR)``
    """
    supported_ccs = get_supported_ccs()

    for i, cc in enumerate(supported_ccs):
        if cc == mycc:
            # Matches
            return cc
        elif cc > mycc:
            # Exceeded
            if i == 0:
                # CC lower than supported
                msg = (
                    "GPU compute capability %d.%d is not supported"
                    "(requires >=%d.%d)" % (mycc + cc)
                )
                raise CCSupportError(msg)
            else:
                # return the previous CC
                return supported_ccs[i - 1]

    # CC higher than supported
    return supported_ccs[-1]  # Choose the highest


def get_arch_option(major, minor):
    """Matches with the closest architecture option"""
    if config.FORCE_CUDA_CC:
        arch = config.FORCE_CUDA_CC
    else:
        arch = find_closest_arch((major, minor))
    return "compute_%d%d" % arch


def get_lowest_supported_cc():
    return min(get_supported_ccs())


def get_supported_ccs():
    if not _HAVE_BINDINGS:
        raise RuntimeError(
            "NVIDIA CUDA bindings are required to query supported archs"
        )
    retcode, archs = bindings_nvrtc.nvrtcGetSupportedArchs()
    if retcode != bindings_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"{retcode.name} when calling nvrtcGetSupportedArchs()"
        )
    return [(arch // 10, arch % 10) for arch in archs]
