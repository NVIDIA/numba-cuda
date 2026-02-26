# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
from collections import namedtuple
from contextlib import contextmanager

from cuda import pathfinder

from numba.cuda import config

import pathlib

_env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])


@contextmanager
def temporary_env_var(key, value):
    """Context manager to temporarily set an environment variable."""
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _get_libdevice_path():
    try:
        located = pathfinder.locate_bitcode_lib("device")
        return _env_path_tuple(located.found_via, located.abs_path)
    except pathfinder.BitcodeLibNotFoundError:
        return _env_path_tuple("<unknown>", None)


def _get_include_dir():
    """Find the root include directory."""
    located_header_dir = pathfinder.locate_nvidia_header_directory("cudart")
    if located_header_dir is not None:
        if not os.path.exists(
            os.path.join(
                located_header_dir.abs_path, "cuda_device_runtime_api.h"
            )
        ):
            return _env_path_tuple("Unknown", None)
        return _env_path_tuple(
            located_header_dir.found_via, located_header_dir.abs_path
        )
    else:
        if config.CUDA_INCLUDE_PATH:
            return _env_path_tuple(
                "CUDA_INCLUDE_PATH Config entry", config.CUDA_INCLUDE_PATH
            )
    return _env_path_tuple("Unknown", None)


def _find_cuda_home_from_lib_path(lib_path):
    """
    Walk up from a library path to find a directory containing 'nvvm' subdirectory.

    For example, given /usr/local/cuda/lib64/libnvrtc.so.12,
    this would find /usr/local/cuda (which contains nvvm/).

    Returns the path if found, None otherwise.
    """
    current = pathlib.Path(lib_path).resolve()

    for parent in current.parents:
        nvvm_subdir = parent / "nvvm"
        if nvvm_subdir.is_dir():
            return str(parent)

    return None


def _get_nvvm():
    # Strategy:
    # 1. Try pathfinder directly
    # 2. If CUDA_HOME/CUDA_PATH are set, pathfinder would have found it - give up
    # 3. Use nvrtc's location to infer CUDA installation root
    # 4. Temporarily set CUDA_HOME and retry pathfinder
    try:
        return pathfinder.load_nvidia_dynamic_lib("nvvm")
    except pathfinder.DynamicLibNotFoundError as e:
        nvvm_exc = e

    def _raise_original(reason: str) -> None:
        raise pathfinder.DynamicLibNotFoundError(
            f"{reason}; original nvvm error: {nvvm_exc}"
        ) from nvvm_exc

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        _raise_original("nvvm not found and CUDA_HOME/CUDA_PATH is set")

    try:
        loaded_nvrtc = _get_nvrtc()
    except Exception as nvrtc_exc:
        raise pathfinder.DynamicLibNotFoundError(
            f"nvrtc load failed while inferring CUDA_HOME; original nvvm error: {nvvm_exc}"
        ) from nvrtc_exc

    if loaded_nvrtc.found_via != "system-search":
        _raise_original(
            f"nvrtc found via {loaded_nvrtc.found_via}, cannot infer CUDA_HOME"
        )

    cuda_home = _find_cuda_home_from_lib_path(loaded_nvrtc.abs_path)
    if cuda_home is None:
        _raise_original(
            f"nvrtc path did not map to CUDA_HOME ({loaded_nvrtc.abs_path})"
        )

    with temporary_env_var("CUDA_HOME", cuda_home):
        try:
            library = pathfinder.load_nvidia_dynamic_lib("nvvm")
        except pathfinder.DynamicLibNotFoundError as exc:
            raise pathfinder.DynamicLibNotFoundError(
                f"nvvm not found after inferring CUDA_HOME={cuda_home}; "
                f"original nvvm error: {nvvm_exc}"
            ) from exc
        library.found_via = "system-search"
        return library


def _get_nvrtc():
    return pathfinder.load_nvidia_dynamic_lib("nvrtc")


def _get_nvrtc_path():
    nvrtc = _get_nvrtc()
    return _env_path_tuple(nvrtc.found_via, nvrtc.abs_path)


def _get_nvvm_path():
    nvvm = _get_nvvm()
    return _env_path_tuple(nvvm.found_via, nvvm.abs_path)


def _get_static_cudalib_path(name):
    """Find a static library via pathfinder."""
    try:
        located = pathfinder.locate_static_lib(name)
        return _env_path_tuple(located.found_via, located.abs_path)
    except pathfinder.StaticLibNotFoundError:
        return _env_path_tuple("<unknown>", None)


def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvrtc": file_path
    - "nvvm": file_path
    - "libdevice": file_path
    - "cudadevrt": file_path
    - "include_dir": directory_path

    Note: The result of the function is cached.
    """
    if hasattr(get_cuda_paths, "_cached_result"):
        return get_cuda_paths._cached_result
    else:
        d = {
            "nvrtc": _get_nvrtc_path(),
            "nvvm": _get_nvvm_path(),
            "libdevice": _get_libdevice_path(),
            "cudadevrt": _get_static_cudalib_path("cudadevrt"),
            "include_dir": _get_include_dir(),
        }
        get_cuda_paths._cached_result = d
        return d
