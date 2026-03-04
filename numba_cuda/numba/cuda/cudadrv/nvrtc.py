# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.cudadrv.error import (
    CCSupportError,
)
from numba.cuda import config
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.utils import _readenv
from cuda import pathfinder
import os
import warnings
import functools

from cuda.core import Program, ProgramOptions
from cuda.bindings import nvrtc as bindings_nvrtc

NVRTC_EXTRA_SEARCH_PATHS = _readenv(
    "NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS", str, ""
) or getattr(config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS", "")
if not hasattr(config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS"):
    config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = NVRTC_EXTRA_SEARCH_PATHS


@functools.cache
def _get_nvrtc_version():
    retcode, major, minor = bindings_nvrtc.nvrtcVersion()
    if retcode != bindings_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"{retcode.name} when calling nvrtcVersion()")
    return (major, minor)


def _verify_cc_tuple(cc):
    version = _get_nvrtc_version()
    ver_str = lambda version: ".".join(str(v) for v in version)

    if len(cc) == 3:
        cc, arch = (cc[0], cc[1]), cc[2]
    else:
        arch = ""

    if arch not in ("", "a", "f"):
        raise ValueError(
            f"Invalid architecture suffix '{arch}' in compute capability "
            f"{ver_str(cc)}{arch}. Expected '', 'a', or 'f'."
        )

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
        found = (found[0], found[1], arch)
        warnings.warn(
            f"Device compute capability {ver_str(cc)} is not supported by "
            f"NVRTC {ver_str(version)}. Using {ver_str(found)} instead."
        )
    else:
        found = (cc[0], cc[1], arch)

    return found


def compile(src, name, cc, ltoir=False, lineinfo=False, debug=False):
    """
    Compile a CUDA C/C++ source to PTX or LTOIR for a given compute capability.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param cc: A tuple ``(major, minor)`` or ``(major, minor, arch)`` of the
        compute capability
    :type cc: tuple
    :param ltoir: Compile into LTOIR if True, otherwise into PTX
    :type ltoir: bool
    :param lineinfo: Whether to include line information in the compiled code
    :type lineinfo: bool
    :param debug: Whether to include debug information in the compiled code
    :type debug: bool
    :return: The compiled PTX or LTOIR and compilation log
    :rtype: tuple
    """
    found = _verify_cc_tuple(cc)
    version = _get_nvrtc_version()

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (rdc) is needed to prevent device functions
    #   being optimized away.
    major, minor = found[0], found[1]
    cc_arch = found[2] if len(found) == 3 else ""

    arch = f"sm_{major}{minor}{cc_arch}"

    cuda_include_dir = get_cuda_paths()["include_dir"].info
    cuda_includes = [f"{cuda_include_dir}"]

    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
    numba_cuda_path = os.path.dirname(cudadrv_path)

    nvrtc_ver_major = version[0]

    if nvrtc_ver_major == 12:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '12')}"

    elif nvrtc_ver_major == 13:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '13')}"

    cccl_found_header_dir = pathfinder.locate_nvidia_header_directory("cccl")
    if cccl_found_header_dir is not None:
        # TODO: Not every kernel needs cccl, so it shouldn't
        # be added to the include path for every kernel.
        # Needs to be flagged during compilation if NRT is included.
        cccl_include_dir = cccl_found_header_dir.abs_path
        cuda_includes.append(cccl_include_dir)

    if config.CUDA_NVRTC_EXTRA_SEARCH_PATHS:
        extra_includes = config.CUDA_NVRTC_EXTRA_SEARCH_PATHS.split(":")
    else:
        extra_includes = []

    nrt_include = os.path.join(numba_cuda_path, "memory_management")

    includes = [numba_include, *cuda_includes, nrt_include, *extra_includes]

    # TODO: move all this into Program/ProgramOptions
    # logsz = config.CUDA_LOG_SIZE
    #
    # jitinfo = bytearray(logsz)
    # jiterrors = bytearray(logsz)
    #
    # jit_option = binding.CUjit_option
    # options = {
    #     jit_option.CU_JIT_INFO_LOG_BUFFER: jitinfo,
    #     jit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: logsz,
    #     jit_option.CU_JIT_ERROR_LOG_BUFFER: jiterrors,
    #     jit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: logsz,
    #     jit_option.CU_JIT_LOG_VERBOSE: config.CUDA_VERBOSE_JIT_LOG,
    # }
    # info_log = jitinfo.decode("utf-8")

    options = ProgramOptions(
        arch=arch,
        include_path=includes,
        relocatable_device_code=True,
        link_time_optimization=ltoir,
        name=name,
        debug=debug,
        lineinfo=lineinfo,
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


def find_closest_arch(cc):
    """
    Given a compute capability, return the closest compute capability supported
    by the CUDA toolkit.

    :param mycc: Compute capability as a tuple ``(MAJOR, MINOR)``
    :return: Closest supported CC as a tuple ``(MAJOR, MINOR)``
    """
    supported_ccs = get_supported_ccs()

    for i, supported_cc in enumerate(supported_ccs):
        if supported_cc == cc:
            # Matches
            return supported_cc
        elif supported_cc > cc:
            # Exceeded
            if i == 0:
                # CC lower than supported
                msg = (
                    "GPU compute capability %d.%d is not supported"
                    "(requires >=%d.%d)" % (cc + supported_cc)
                )
                raise CCSupportError(msg)
            else:
                # return the previous CC
                return supported_ccs[i - 1]

    # CC higher than supported
    return supported_ccs[-1]  # Choose the highest


def get_arch_option(major, minor, arch=""):
    """Matches with the closest architecture option"""
    if config.FORCE_CUDA_CC:
        fcc = config.FORCE_CUDA_CC
        major, minor = fcc[0], fcc[1]
        if len(fcc) == 3:
            arch = fcc[2]
        else:
            arch = ""
    else:
        new_major, new_minor = find_closest_arch((major, minor))
        if (new_major, new_minor) != (major, minor):
            # If we picked a different major / minor, then using an
            # arch-specific version is invalid
            if arch != "":
                raise ValueError(
                    f"Can't use arch-specific compute_{major}{minor}{arch} with "
                    "closest found compute capability "
                    f"compute_{new_major}{new_minor}"
                )
        major, minor = new_major, new_minor

    return f"compute_{major}{minor}{arch}"


def get_lowest_supported_cc():
    return min(get_supported_ccs())


def get_supported_ccs():
    retcode, archs = bindings_nvrtc.nvrtcGetSupportedArchs()
    if retcode != bindings_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"{retcode.name} when calling nvrtcGetSupportedArchs()"
        )
    return [(arch // 10, arch % 10) for arch in archs]
