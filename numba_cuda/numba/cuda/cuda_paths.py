# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
import os
from collections import namedtuple
import platform
import importlib.metadata
from numba.cuda.core.config import IS_WIN32
from numba.cuda.misc.findlib import find_lib
from numba.cuda import config
from cuda import pathfinder
import pathlib
from contextlib import contextmanager

_env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])

SEARCH_PRIORITY = [
    "Conda environment",
    "NVIDIA NVCC Wheel",
    "CUDA_HOME",
    "System",
]


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


def _get_distribution(distribution_name):
    """Get the distribution path using importlib.metadata, returning None if not found."""
    try:
        dist = importlib.metadata.distribution(distribution_name)
        return dist
    except importlib.metadata.PackageNotFoundError:
        return None


def _priority_index(label):
    if label in SEARCH_PRIORITY:
        return SEARCH_PRIORITY.index(label)
    else:
        raise ValueError(f"Can't determine search priority for {label}")


def _find_first_valid_lazy(options):
    sorted_options = sorted(options, key=lambda x: _priority_index(x[0]))
    for label, fn in sorted_options:
        value = fn()
        if value:
            return label, value
    return "<unknown>", None


def _build_options(pairs):
    """Sorts and returns a list of (label, value) tuples according to SEARCH_PRIORITY."""
    priority_index = {label: i for i, label in enumerate(SEARCH_PRIORITY)}
    return sorted(
        pairs, key=lambda pair: priority_index.get(pair[0], float("inf"))
    )


def _find_valid_path(options):
    """Find valid path from *options*, which is a list of 2-tuple of
    (name, path).  Return first pair where *path* is not None.
    If no valid path is found, return ('<unknown>', None)
    """
    for by, data in options:
        if data is not None:
            return by, data
    else:
        return "<unknown>", None


def _get_libdevice_path_decision():
    options = _build_options(
        [
            ("Conda environment", get_libdevice_conda_path),
            ("NVIDIA NVCC Wheel", get_libdevice_wheel_path),
            (
                "CUDA_HOME",
                lambda: get_cuda_home("nvvm", "libdevice", "libdevice.10.bc"),
            ),
            (
                "System",
                lambda: get_system_ctk("nvvm", "libdevice", "libdevice.10.bc"),
            ),
        ]
    )
    return _find_first_valid_lazy(options)


def _get_libdevice_path():
    by, out = _get_libdevice_path_decision()
    if not out:
        return _env_path_tuple(by, None)
    return _env_path_tuple(by, out)


def _cuda_static_libdir():
    if IS_WIN32:
        return ("lib", "x64")
    else:
        return ("lib64",)


def _get_cudalib_wheel_libdir():
    """Get the cudalib path from the cudart wheel."""
    cuda_module_lib_dir = None
    cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime-cu12")
    if cuda_runtime_distribution is not None:
        site_packages_path = cuda_runtime_distribution.locate_file("")
        cuda_module_lib_dir = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_runtime",
            "bin" if IS_WIN32 else "lib",
        )
    else:
        cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime")
        if (
            cuda_runtime_distribution is not None
            and cuda_runtime_distribution.version.startswith("13.")
        ):
            site_packages_path = cuda_runtime_distribution.locate_file("")
            cuda_module_lib_dir = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "bin" if IS_WIN32 else "lib",
                "x86_64" if IS_WIN32 else "",
            )

    if cuda_module_lib_dir is None:
        return None

    if cuda_module_lib_dir and os.path.isdir(cuda_module_lib_dir):
        return cuda_module_lib_dir
    return None


def _get_cudalib_dir_path_decision():
    options = _build_options(
        [
            ("Conda environment", get_conda_ctk_libdir),
            ("NVIDIA NVCC Wheel", _get_cudalib_wheel_libdir),
            ("CUDA_HOME", get_cuda_home_libdir),
            ("System", get_system_ctk_libdir),
        ]
    )
    return _find_first_valid_lazy(options)


def _get_static_cudalib_dir_path_decision():
    options = _build_options(
        [
            ("Conda environment", get_conda_ctk_libdir),
            ("NVIDIA NVCC Wheel", get_wheel_static_libdir),
            (
                "CUDA_HOME",
                lambda: get_cuda_home(*_cuda_static_libdir()),
            ),
            ("System", lambda: get_system_ctk(*_cuda_static_libdir())),
        ]
    )
    return _find_first_valid_lazy(options)


def _get_cudalib_dir():
    by, libdir = _get_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def _get_static_cudalib_dir():
    by, libdir = _get_static_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def get_system_ctk(*subdirs):
    """Return path to system-wide cudatoolkit; or, None if it doesn't exist."""
    # Linux?
    if not IS_WIN32:
        # Is cuda alias to /usr/local/cuda?
        # We are intentionally not getting versioned cuda installation.
        result = os.path.join("/usr/local/cuda", *subdirs)
        if os.path.exists(result):
            return result
        return None
    return None


def get_system_ctk_libdir():
    """Return path to directory containing the shared libraries of cudatoolkit."""
    system_ctk_dir = get_system_ctk()
    if system_ctk_dir is None:
        return None
    libdir = os.path.join(
        system_ctk_dir,
        "Library" if IS_WIN32 else "lib64",
        "bin" if IS_WIN32 else "",
    )
    # Windows CUDA 13 system CTK uses "bin\x64" directory
    if IS_WIN32 and os.path.isdir(os.path.join(libdir, "x64")):
        libdir = os.path.join(libdir, "x64")

    if libdir and os.path.isdir(libdir):
        return os.path.normpath(libdir)
    return None


def get_system_ctk_include():
    system_ctk_dir = get_system_ctk()
    if system_ctk_dir is None:
        return None
    include_dir = os.path.join(system_ctk_dir, "include")

    if include_dir and os.path.isdir(include_dir):
        if os.path.isfile(
            os.path.join(include_dir, "cuda_device_runtime_api.h")
        ):
            return include_dir
    return None


def get_conda_ctk_libdir():
    """Return path to directory containing the shared libraries of cudatoolkit."""
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return None
    libdir = os.path.join(
        sys.prefix,
        "Library" if IS_WIN32 else "lib",
        "bin" if IS_WIN32 else "",
    )
    # Windows CUDA 13.0.0 uses "bin\x64" directory but 13.0.1+ just uses "bin" directory
    if IS_WIN32 and os.path.isdir(os.path.join(libdir, "x64")):
        libdir = os.path.join(libdir, "x64")
    # Assume the existence of nvrtc to imply needed CTK libraries are installed
    paths = find_lib("nvrtc", libdir)
    if not paths:
        return None
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_libdevice_conda_path():
    """Return path to directory containing the libdevice bitcode library."""
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return None

    # Linux: nvvm/libdevice/libdevice.10.bc
    # Windows: Library/nvvm/libdevice/libdevice.10.bc
    libdevice_path = os.path.join(
        sys.prefix,
        "Library" if IS_WIN32 else "",
        "nvvm",
        "libdevice",
        "libdevice.10.bc",
    )
    if os.path.isfile(libdevice_path):
        return libdevice_path
    return None


def get_wheel_static_libdir():
    cuda_module_static_lib_dir = None
    # CUDA 12
    cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime-cu12")
    if cuda_runtime_distribution is not None:
        site_packages_path = cuda_runtime_distribution.locate_file("")
        cuda_module_static_lib_dir = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_runtime",
            "lib",
            "x64" if IS_WIN32 else "",
        )
    else:
        cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime")
        if (
            cuda_runtime_distribution is not None
            and cuda_runtime_distribution.version.startswith("13.")
        ):
            site_packages_path = cuda_runtime_distribution.locate_file("")
            cuda_module_static_lib_dir = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "lib",
                "x64" if IS_WIN32 else "",
            )

    if cuda_module_static_lib_dir is None:
        return None

    cudadevrt_path = os.path.join(
        cuda_module_static_lib_dir,
        "cudadevrt.lib" if IS_WIN32 else "libcudadevrt.a",
    )

    if cudadevrt_path and os.path.isfile(cudadevrt_path):
        return os.path.dirname(cudadevrt_path)
    return None


def get_wheel_include():
    cuda_module_include_dir = None
    # CUDA 12
    cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime-cu12")
    if cuda_runtime_distribution is not None:
        site_packages_path = cuda_runtime_distribution.locate_file("")
        cuda_module_include_dir = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_runtime",
            "include",
        )
    else:
        cuda_runtime_distribution = _get_distribution("nvidia-cuda-runtime")
        if (
            cuda_runtime_distribution is not None
            and cuda_runtime_distribution.version.startswith("13.")
        ):
            site_packages_path = cuda_runtime_distribution.locate_file("")
            cuda_module_include_dir = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "include",
            )

    if cuda_module_include_dir and os.path.isdir(cuda_module_include_dir):
        if os.path.isfile(
            os.path.join(cuda_module_include_dir, "cuda_device_runtime_api.h")
        ):
            return cuda_module_include_dir
    return None


def get_cuda_home(*subdirs):
    """Get paths of CUDA_HOME.
    If *subdirs* are the subdirectory name to be appended in the resulting
    path.
    """
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        # Try Windows CUDA installation without Anaconda
        cuda_home = os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)
    return None


def get_cuda_home_libdir():
    """Return path to directory containing the shared libraries of cudatoolkit."""
    cuda_home_dir = get_cuda_home()
    if cuda_home_dir is None:
        return None
    libdir = os.path.join(
        cuda_home_dir,
        "Library" if IS_WIN32 else "lib64",
        "bin" if IS_WIN32 else "",
    )
    # Windows CUDA 13 system CTK uses "bin\x64" directory while conda just uses "bin" directory
    if IS_WIN32 and os.path.isdir(os.path.join(libdir, "x64")):
        libdir = os.path.join(libdir, "x64")
    return os.path.normpath(libdir)


def get_cuda_home_include():
    cuda_home_dir = get_cuda_home()
    if cuda_home_dir is None:
        return None
    include_dir = cuda_home_dir
    # For Windows, CTK puts it in $CTK/include but conda puts it in $CTK/Library/include
    if IS_WIN32:
        if os.path.isdir(os.path.join(include_dir, "Library")):
            include_dir = os.path.join(include_dir, "Library", "include")
        else:
            include_dir = os.path.join(include_dir, "include")
    else:
        include_dir = os.path.join(include_dir, "include")

    if include_dir and os.path.isdir(include_dir):
        if os.path.isfile(
            os.path.join(include_dir, "cuda_device_runtime_api.h")
        ):
            return include_dir
    return None


def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvrtc": file_path
    - "nvvm": file_path
    - "libdevice": file_path
    - "cudalib_dir": directory_path
    - "static_cudalib_dir": directory_path
    - "include_dir": directory_path

    Note: The result of the function is cached.
    """
    # Check cache
    if hasattr(get_cuda_paths, "_cached_result"):
        return get_cuda_paths._cached_result
    else:
        # Not in cache
        d = {
            "nvrtc": _get_nvrtc_path(),
            "nvvm": _get_nvvm_path(),
            "libdevice": _get_libdevice_path(),
            "cudalib_dir": _get_cudalib_dir(),
            "static_cudalib_dir": _get_static_cudalib_dir(),
            "include_dir": _get_include_dir(),
        }
        # Cache result
        get_cuda_paths._cached_result = d
        return d


def get_libdevice_wheel_path():
    libdevice_path = None
    # CUDA 12
    nvvm_distribution = _get_distribution("nvidia-cuda-nvcc-cu12")
    if nvvm_distribution is not None:
        site_packages_path = nvvm_distribution.locate_file("")
        libdevice_path = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_nvcc",
            "nvvm",
            "libdevice",
            "libdevice.10.bc",
        )

    # CUDA 13
    if libdevice_path is None:
        nvvm_distribution = _get_distribution("nvidia-nvvm")
        if (
            nvvm_distribution is not None
            and nvvm_distribution.version.startswith("13.")
        ):
            site_packages_path = nvvm_distribution.locate_file("")
            libdevice_path = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "nvvm",
                "libdevice",
                "libdevice.10.bc",
            )

    if libdevice_path and os.path.isfile(libdevice_path):
        return libdevice_path
    return None


def get_current_cuda_target_name():
    """Determine conda's CTK target folder based on system and machine arch.

    CTK's conda package delivers headers based on its architecture type. For example,
    `x86_64` machine places header under `$CONDA_PREFIX/targets/x86_64-linux`, and
    `aarch64` places under `$CONDA_PREFIX/targets/sbsa-linux`. Read more about the
    nuances at cudart's conda feedstock:
    https://github.com/conda-forge/cuda-cudart-feedstock/blob/main/recipe/meta.yaml#L8-L11  # noqa: E501
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Linux":
        arch_to_targets = {"x86_64": "x86_64-linux", "aarch64": "sbsa-linux"}
    elif system == "Windows":
        arch_to_targets = {
            "AMD64": "x64",
        }
    else:
        arch_to_targets = {}

    return arch_to_targets.get(machine, None)


def get_conda_include_dir():
    """
    Return the include directory in the current conda environment, if one
    is active and it exists.
    """
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return

    if IS_WIN32:
        include_dir = os.path.join(sys.prefix, "Library", "include")
    elif target_name := get_current_cuda_target_name():
        include_dir = os.path.join(
            sys.prefix, "targets", target_name, "include"
        )
    else:
        # A fallback when target cannot determined
        # though usually it shouldn't.
        include_dir = os.path.join(sys.prefix, "include")

    if os.path.isdir(include_dir) and os.path.isfile(
        os.path.join(include_dir, "cuda_device_runtime_api.h")
    ):
        return include_dir
    return None


def _get_include_dir():
    """Find the root include directory."""
    options = [
        ("Conda environment (NVIDIA package)", get_conda_include_dir()),
        ("NVIDIA NVCC Wheel", get_wheel_include()),
        ("CUDA_HOME", get_cuda_home_include()),
        ("System", get_system_ctk_include()),
        ("CUDA_INCLUDE_PATH Config Entry", config.CUDA_INCLUDE_PATH),
    ]
    by, include_dir = _find_valid_path(options)
    return _env_path_tuple(by, include_dir)


def _find_cuda_home_from_lib_path(lib_path):
    """
    Walk up from a library path to find a directory containing 'nvvm' subdirectory.

    For example, given /usr/local/cuda/lib64/libnvrtc.so.12,
    this would find /usr/local/cuda (which contains nvvm/).

    Returns the path if found, None otherwise.
    """
    current = pathlib.Path(lib_path).resolve()

    # Walk up the directory tree
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
    # First, try pathfinder directly
    try:
        return pathfinder.load_nvidia_dynamic_lib("nvvm")
    except pathfinder.DynamicLibNotFoundError as e:
        nvvm_exc = e

    def _raise_original(reason: str) -> None:
        raise pathfinder.DynamicLibNotFoundError(
            f"{reason}; original nvvm error: {nvvm_exc}"
        ) from nvvm_exc

    # If CUDA_HOME or CUDA_PATH is set, pathfinder would have found libnvvm
    # based on the environment variable(s) - nothing more we can do
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        _raise_original("nvvm not found and CUDA_HOME/CUDA_PATH is set")
    # Try to locate nvrtc - this library is almost certainly needed if nvvm is needed (in the context of numba-cuda)
    try:
        loaded_nvrtc = _get_nvrtc()
    except Exception as nvrtc_exc:
        raise pathfinder.DynamicLibNotFoundError(
            f"nvrtc load failed while inferring CUDA_HOME; original nvvm error: {nvvm_exc}"
        ) from nvrtc_exc
    # If nvrtc was not found via system-search, we can't reliably determine
    # the CUDA installation structure
    if loaded_nvrtc.found_via != "system-search":
        _raise_original(
            f"nvrtc found via {loaded_nvrtc.found_via}, cannot infer CUDA_HOME"
        )
    # Search backward from nvrtc's location to find a directory with "nvvm" subdirectory
    cuda_home = _find_cuda_home_from_lib_path(loaded_nvrtc.abs_path)
    if cuda_home is None:
        _raise_original(
            f"nvrtc path did not map to CUDA_HOME ({loaded_nvrtc.abs_path})"
        )
    # Temporarily set CUDA_HOME and retry pathfinder
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
    # the pathfinder API will either find the library or raise
    nvrtc = _get_nvrtc()
    return _env_path_tuple(nvrtc.found_via, nvrtc.abs_path)


def _get_nvvm_path():
    # the pathfinder API will either find the library or raise
    nvvm = _get_nvvm()
    return _env_path_tuple(nvvm.found_via, nvvm.abs_path)
