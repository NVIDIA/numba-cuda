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

_env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])

SEARCH_PRIORITY = [
    "Conda environment",
    "NVIDIA NVCC Wheel",
    "CUDA_HOME",
    "System",
]


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


def _get_nvvm_path_decision():
    options = _build_options(
        [
            ("Conda environment", _get_nvvm_conda_path),
            ("NVIDIA NVCC Wheel", _get_nvvm_wheel_path),
            ("CUDA_HOME", _get_nvvm_cuda_home_path),
            ("System", _get_nvvm_system_path),
        ]
    )
    return _find_first_valid_lazy(options)


def _get_nvrtc_path_decision():
    options = _build_options(
        [
            ("Conda environment", get_conda_ctk_libdir),
            ("NVIDIA NVCC Wheel", _get_nvrtc_wheel_libdir),
            ("CUDA_HOME", get_cuda_home_libdir),
            ("System", get_system_ctk_libdir),
        ]
    )
    return _find_first_valid_lazy(options)


def _get_nvvm_wheel_path():
    dso_path = None
    # CUDA 12
    nvcc_distribution = _get_distribution("nvidia-cuda-nvcc-cu12")
    if nvcc_distribution is not None:
        site_packages_path = nvcc_distribution.locate_file("")
        nvvm_lib_dir = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_nvcc",
            "nvvm",
            "bin" if IS_WIN32 else "lib64",
        )
        dso_path = os.path.join(
            nvvm_lib_dir, "nvvm64_40_0.dll" if IS_WIN32 else "libnvvm.so"
        )

    # CUDA 13
    if dso_path is None:
        nvcc_distribution = _get_distribution("nvidia-nvvm")
        if (
            nvcc_distribution is not None
            and nvcc_distribution.version.startswith("13.")
        ):
            site_packages_path = nvcc_distribution.locate_file("")
            nvvm_lib_dir = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "bin" if IS_WIN32 else "lib",
                "x86_64" if IS_WIN32 else "",
            )
            dso_path = os.path.join(
                nvvm_lib_dir, "nvvm64_40_0.dll" if IS_WIN32 else "libnvvm.so.4"
            )

    if dso_path and os.path.isfile(dso_path):
        return dso_path
    return None


def _get_nvrtc_wheel_libdir():
    dso_path = None
    # CUDA 12
    nvrtc_distribution = _get_distribution("nvidia-cuda-nvrtc-cu12")
    if nvrtc_distribution is not None:
        site_packages_path = nvrtc_distribution.locate_file("")
        nvrtc_lib_dir = os.path.join(
            site_packages_path,
            "nvidia",
            "cuda_nvrtc",
            "bin" if IS_WIN32 else "lib",
        )
        dso_path = os.path.join(
            nvrtc_lib_dir, "nvrtc64_120_0.dll" if IS_WIN32 else "libnvrtc.so.12"
        )

    # CUDA 13
    if dso_path is None:
        nvrtc_distribution = _get_distribution("nvidia-cuda-nvrtc")
        if (
            nvrtc_distribution is not None
            and nvrtc_distribution.version.startswith("13.")
        ):
            site_packages_path = nvrtc_distribution.locate_file("")
            nvrtc_lib_dir = os.path.join(
                site_packages_path,
                "nvidia",
                "cu13",
                "bin" if IS_WIN32 else "lib",
                "x86_64" if IS_WIN32 else "",
            )
            dso_path = os.path.join(
                nvrtc_lib_dir,
                "nvrtc64_130_0.dll" if IS_WIN32 else "libnvrtc.so.13",
            )

    if dso_path and os.path.isfile(dso_path):
        return os.path.dirname(dso_path)
    return None


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


def _get_nvvm_system_path():
    nvvm_lib_dir = get_system_ctk("nvvm")
    if nvvm_lib_dir is None:
        return None
    nvvm_lib_dir = os.path.join(nvvm_lib_dir, "bin" if IS_WIN32 else "lib64")
    if IS_WIN32 and os.path.isdir(os.path.join(nvvm_lib_dir, "x64")):
        nvvm_lib_dir = os.path.join(nvvm_lib_dir, "x64")

    nvvm_path = os.path.join(
        nvvm_lib_dir, "nvvm64_40_0.dll" if IS_WIN32 else "libnvvm.so.4"
    )
    # if os.path.isfile(nvvm_path):
    #     return nvvm_path
    return nvvm_path


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


def _get_nvvm_conda_path():
    """Return path to directory containing the nvvm library."""
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
    if not is_conda_env:
        return None
    nvvm_dir = os.path.join(
        sys.prefix,
        "Library" if IS_WIN32 else "",
        "nvvm",
        "bin" if IS_WIN32 else "lib64",
    )
    # Windows CUDA 13.0.0 puts in "bin\x64" directory but 13.0.1+ just uses "bin" directory
    if IS_WIN32 and os.path.isdir(os.path.join(nvvm_dir, "x64")):
        nvvm_dir = os.path.join(nvvm_dir, "x64")

    nvvm_path = os.path.join(
        nvvm_dir, "nvvm64_40_0.dll" if IS_WIN32 else "libnvvm.so.4"
    )
    if os.path.isfile(nvvm_path):
        return nvvm_path
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


def _get_nvvm_cuda_home_path():
    nvvm_lib_dir = get_cuda_home("nvvm")
    if nvvm_lib_dir is None:
        return
    nvvm_lib_dir = os.path.join(nvvm_lib_dir, "bin" if IS_WIN32 else "lib64")
    if IS_WIN32 and os.path.isdir(os.path.join(nvvm_lib_dir, "x64")):
        nvvm_lib_dir = os.path.join(nvvm_lib_dir, "x64")

    nvvm_path = os.path.join(
        nvvm_lib_dir, "nvvm64_40_0.dll" if IS_WIN32 else "libnvvm.so.4"
    )
    # if os.path.isfile(nvvm_path):
    #     return nvvm_path
    return nvvm_path


def _get_nvvm_path():
    by, out = _get_nvvm_path_decision()
    if not out:
        return _env_path_tuple(by, None)
    return _env_path_tuple(by, out)


def _get_nvrtc_path():
    by, path = _get_nvrtc_path_decision()
    candidates = find_lib("nvrtc", libdir=path)
    path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)


def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvvm": file_path
    - "nvrtc": file_path
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
            "nvvm": _get_nvvm_path(),
            "nvrtc": _get_nvrtc_path(),
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
