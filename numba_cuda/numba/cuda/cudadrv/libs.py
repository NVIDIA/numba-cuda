# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""CUDA Toolkit libraries lookup utilities.

CUDA Toolkit libraries can be available via either:

- the `cuda-nvcc` and `cuda-nvrtc` conda packages,
- a user supplied location from CUDA_HOME,
- a system wide location,
- package-specific locations (e.g. the Debian NVIDIA packages),
- or can be discovered by the system loader.
"""

import os
import sys
import ctypes

from numba.cuda.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.core import config


if sys.platform == "win32":
    _dllnamepattern = "%s.dll"
    _staticnamepattern = "%s.lib"
elif sys.platform == "darwin":
    _dllnamepattern = "lib%s.dylib"
    _staticnamepattern = "lib%s.a"
else:
    _dllnamepattern = "lib%s.so"
    _staticnamepattern = "lib%s.a"


def get_libdevice():
    d = get_cuda_paths()
    paths = d["libdevice"].info
    return paths


def open_libdevice():
    with open(get_libdevice(), "rb") as bcfile:
        return bcfile.read()


def get_cudalib(lib, static=False):
    """
    Find the path of a CUDA library based on a search of known locations. If
    the search fails, return a generic filename for the library (e.g.
    'libnvvm.so' for 'nvvm') so that we may attempt to load it using the system
    loader's search mechanism.
    """
    if lib in {"nvrtc", "nvvm"}:
        return get_cuda_paths()[lib].info or _dllnamepattern % lib

    dir_type = "static_cudalib_dir" if static else "cudalib_dir"
    libdir = get_cuda_paths()[dir_type].info

    candidates = find_lib(lib, libdir, static=static)
    namepattern = _staticnamepattern if static else _dllnamepattern
    return max(candidates) if candidates else namepattern % lib


def get_cuda_include_dir():
    """
    Find the path to cuda include dir based on a list of default locations.
    Note that this does not list the `CUDA_INCLUDE_PATH` entry in user
    configuration.
    """

    return get_cuda_paths()["include_dir"].info


def check_cuda_include_dir(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    if not os.path.exists(os.path.join(path, "cuda_runtime.h")):
        raise FileNotFoundError(f"Unable to find cuda_runtime.h from {path}")


def open_cudalib(lib):
    path = get_cudalib(lib)
    return ctypes.CDLL(path)


def check_static_lib(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")


def _get_source_variable(lib, static=False):
    if lib == "nvvm":
        return get_cuda_paths()["nvvm"].by
    elif lib == "nvrtc":
        return get_cuda_paths()["nvrtc"].by
    elif lib == "libdevice":
        return get_cuda_paths()["libdevice"].by
    elif lib == "include_dir":
        return get_cuda_paths()["include_dir"].by
    else:
        dir_type = "static_cudalib_dir" if static else "cudalib_dir"
        return get_cuda_paths()[dir_type].by


def test():
    """Test library lookup.  Path info is printed to stdout."""
    failed = False

    # Check for the driver
    try:
        dlloader, candidates = locate_driver_and_loader()
        print("Finding driver from candidates:")
        for location in candidates:
            print(f"\t{location}")
        print(f"Using loader {dlloader}")
        print("\tTrying to load driver", end="...")
        dll, path = load_driver(dlloader, candidates)
        print("\tok")
        print(f"\t\tLoaded from {path}")
    except CudaSupportError as e:
        print(f"\tERROR: failed to open driver: {e}")
        failed = True

    # Find the absolute location of the driver on Linux. Various driver-related
    # issues have been reported by WSL2 users, and it is almost always due to a
    # Linux (i.e. not- WSL2) driver being installed in a WSL2 system.
    # Providing the absolute location of the driver indicates its version
    # number in the soname (e.g. "libcuda.so.530.30.02"), which can be used to
    # look up whether the driver was intended for "native" Linux.
    if sys.platform == "linux" and not failed:
        pid = os.getpid()
        mapsfile = os.path.join(os.path.sep, "proc", f"{pid}", "maps")
        try:
            with open(mapsfile) as f:
                maps = f.read()
        # It's difficult to predict all that might go wrong reading the maps
        # file - in case various error conditions ensue (the file is not found,
        # not readable, etc.) we use OSError to hopefully catch any of them.
        except OSError:
            # It's helpful to report that this went wrong to the user, but we
            # don't set failed to True because this doesn't have any connection
            # to actual CUDA functionality.
            print(
                f"\tERROR: Could not open {mapsfile} to determine absolute "
                "path to libcuda.so"
            )
        else:
            # In this case we could read the maps, so we can report the
            # relevant ones to the user
            locations = set(s for s in maps.split() if "libcuda.so" in s)
            print("\tMapped libcuda.so paths:")
            for location in locations:
                print(f"\t\t{location}")

    # Checks for dynamic libraries
    libs = "nvvm nvrtc".split()
    for lib in libs:
        path = get_cudalib(lib)
        print("Finding {} from {}".format(lib, _get_source_variable(lib)))
        print("\tLocated at", path)

        try:
            print("\tTrying to open library", end="...")
            open_cudalib(lib)
            print("\tok")
        except OSError as e:
            print("\tERROR: failed to open %s:\n%s" % (lib, e))
            failed = True

    # Check for cudadevrt (the only static library)
    lib = "cudadevrt"
    path = get_cudalib(lib, static=True)
    print(
        "Finding {} from {}".format(lib, _get_source_variable(lib, static=True))
    )
    print("\tLocated at", path)

    try:
        print("\tChecking library", end="...")
        check_static_lib(path)
        print("\tok")
    except FileNotFoundError as e:
        print("\tERROR: failed to find %s:\n%s" % (lib, e))
        failed = True

    # Check for libdevice
    where = _get_source_variable("libdevice")
    print(f"Finding libdevice from {where}")
    path = get_libdevice()
    print("\tLocated at", path)

    try:
        print("\tChecking library", end="...")
        check_static_lib(path)
        print("\tok")
    except FileNotFoundError as e:
        print("\tERROR: failed to find %s:\n%s" % (lib, e))
        failed = True

    # Check cuda include paths

    print("Include directory configuration variable:")
    print(f"\tCUDA_INCLUDE_PATH={config.CUDA_INCLUDE_PATH}")

    where = _get_source_variable("include_dir")
    print(f"Finding include directory from {where}")
    include = get_cuda_include_dir()
    print("\tLocated at", include)
    try:
        print("\tChecking include directory", end="...")
        check_cuda_include_dir(include)
        print("\tok")
    except FileNotFoundError as e:
        print("\tERROR: failed to find cuda include directory:\n%s" % e)
        failed = True

    return not failed
