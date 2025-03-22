import sys
import re
import os
from collections import namedtuple
import platform
import site
from pathlib import Path
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
from numba import config
import glob
import ctypes


_env_path_tuple = namedtuple('_env_path_tuple', ['by', 'info'])


def _find_valid_path(options):
    """Find valid path from *options*, which is a list of 2-tuple of
    (name, path).  Return first pair where *path* is not None.
    If no valid path is found, return ('<unknown>', None)
    """
    for by, data in options:
        if data is not None:
            return by, data
    else:
        return '<unknown>', None


def _get_libdevice_path_decision():
    options = [
        ('Conda environment', get_conda_ctk()),
        ('Conda environment (NVIDIA package)', get_nvidia_libdevice_ctk()),
        ('CUDA_HOME', get_cuda_home('nvvm', 'libdevice')),
        ('System', get_system_ctk('nvvm', 'libdevice')),
        ('NVIDIA NVCC Wheel', get_libdevice_wheel()),
        ('Debian package', get_debian_pkg_libdevice()),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _nvvm_lib_dir():
    if IS_WIN32:
        return 'nvvm', 'bin'
    else:
        return 'nvvm', 'lib64'


def _get_nvvm_path_decision():
    options = [
        ('Conda environment', get_conda_ctk()),
        ('Conda environment (NVIDIA package)', get_nvidia_nvvm_ctk()),
        ('CUDA_HOME', get_cuda_home(*_nvvm_lib_dir())),
        ('System', get_system_ctk(*_nvvm_lib_dir())),
        ('NVIDIA NVCC Wheel', _get_nvvm_wheel()),
    ]

    by, path = _find_valid_path(options)
    return by, path


def _get_nvrtc_path_decision():
    options = [
        ('CUDA_HOME', get_cuda_home('nvrtc')),
        ('Conda environment', get_conda_ctk()),
        ('System', get_system_ctk('nvrtc')),
        ('NVIDIA NVCC Wheel', _get_nvrtc_wheel()),
    ]
    breakpoint()
    by, path = _find_valid_path(options)
    return by, path


def _get_nvvm_wheel():
    site_paths = [
        site.getusersitepackages()
    ] + site.getsitepackages()
    # The SONAME is taken based on public CTK 12.x releases
    for sp in site_paths:
        if sys.platform.startswith("linux"):
            dso_dir = "lib64"
            # Hack: libnvvm from Linux wheel
            # does not have any soname (CUDAINST-3183)
            dso_path = "libnvvm.so"
        elif sys.platform.startswith("win32"):
            dso_dir = "bin"
            dso_path = "nvvm64_40_0.dll"
        else:
            raise NotImplementedError('Unsupported platform')
        if sp is not None:
            dso_dir = os.path.join(
                sp,
                "nvidia",
                "cuda_nvcc",
                "nvvm",
                dso_dir
            )
            dso_path = os.path.join(dso_dir, dso_path)
            if os.path.exists(dso_path):
                return str(Path(dso_path).parent)


def detect_nvrtc_major_cuda_version(lib_dir):
    if sys.platform.startswith("linux"):
        pattern = os.path.join(lib_dir, "libnvrtc.so.*")
    elif sys.platform.startswith("win32"):
        pattern = os.path.join(lib_dir, "nvrtc64_*.dll")
    else:
        raise NotImplementedError("Unsupported platform")

    candidates = glob.glob(pattern)
    for lib in candidates:
        match = re.search(
            r"libnvrtc\.so\.(\d+)(?:\.(\d+))?$"
            if sys.platform.startswith("linux")
            else r"nvrtc64_(\d+)(\d)_0", os.path.basename(lib)
        )
        if match:
            major, _ = match.groups()
            return int(major)

    raise RuntimeError("CUDA version could not be detected")


def get_nvrtc_dso_path():
    site_paths = [site.getusersitepackages()] + site.getsitepackages()
    for sp in site_paths:
        lib_dir = os.path.join(
            sp,
            "nvidia",
            "cuda_nvrtc",
            (
                "lib" if sys.platform.startswith("linux") else "bin"
            ) if sp else None
        )
        if lib_dir and os.path.exists(lib_dir):

            try:
                major = detect_nvrtc_major_cuda_version(lib_dir)
                if major == 11:
                    cu_ver = (
                        "11.2" if sys.platform.startswith("linux") else "112"
                    )
                elif major == 12:
                    cu_ver = (
                        "12" if sys.platform.startswith("linux") else "120"
                    )
                else:
                    raise NotImplementedError(f"CUDA {major} is not supported")

                return os.path.join(
                    lib_dir,
                    f"libnvrtc.so.{cu_ver}"
                    if sys.platform.startswith("linux")
                    else f"nvrtc64_{cu_ver}_0.dll"
                )
            except RuntimeError:
                continue


def _get_nvrtc_wheel():
    dso_path = get_nvrtc_dso_path()
    if dso_path:
        try:
            result = ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        else:
            if sys.platform.startswith('win32'):
                import win32api

                # This absolute path will
                # always be correct regardless of the package source
                nvrtc_path = win32api.GetModuleFileNameW(result._handle)
                dso_dir = os.path.dirname(nvrtc_path)
                dso_path = os.path.join(
                    dso_dir,
                    [
                        f for f in os.listdir(dso_dir)
                        if re.match("^nvrtc-builtins.*.dll$", f)
                    ][0]
                )
        return Path(dso_path)


def _get_libdevice_paths():
    by, libdir = _get_libdevice_path_decision()
    out = os.path.join(libdir, "libdevice.10.bc")
    return _env_path_tuple(by, out)


def _cudalib_path():
    if IS_WIN32:
        return 'bin'
    else:
        return 'lib64'


def _cuda_home_static_cudalib_path():
    if IS_WIN32:
        return ('lib', 'x64')
    else:
        return ('lib64',)


def _get_cudalib_dir_path_decision():
    options = [
        ('Conda environment', get_conda_ctk()),
        ('Conda environment (NVIDIA package)', get_nvidia_cudalib_ctk()),
        ('CUDA_HOME', get_cuda_home(_cudalib_path())),
        ('System', get_system_ctk(_cudalib_path())),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_static_cudalib_dir_path_decision():
    options = [
        ('Conda environment', get_conda_ctk()),
        ('Conda environment (NVIDIA package)', get_nvidia_static_cudalib_ctk()),
        ('CUDA_HOME', get_cuda_home(*_cuda_home_static_cudalib_path())),
        ('System', get_system_ctk(_cudalib_path())),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_cudalib_dir():
    by, libdir = _get_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def _get_static_cudalib_dir():
    by, libdir = _get_static_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def get_system_ctk(*subdirs):
    """Return path to system-wide cudatoolkit; or, None if it doesn't exist.
    """
    # Linux?
    if sys.platform.startswith('linux'):
        # Is cuda alias to /usr/local/cuda?
        # We are intentionally not getting versioned cuda installation.
        result = os.path.join('/usr/local/cuda', *subdirs)
        if os.path.exists(result):
            return result


def get_conda_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return
    # Assume the existence of NVVM to imply cudatoolkit installed
    paths = find_lib('nvvm')
    if not paths:
        return
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_nvidia_nvvm_ctk():
    """Return path to directory containing the NVVM shared library.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return

    # Assume the existence of NVVM in the conda env implies that a CUDA toolkit
    # conda package is installed.

    # First, try the location used on Linux and the Windows 11.x packages
    libdir = os.path.join(sys.prefix, 'nvvm', _cudalib_path())
    if not os.path.exists(libdir) or not os.path.isdir(libdir):
        # If that fails, try the location used for Windows 12.x packages
        libdir = os.path.join(sys.prefix, 'Library', 'nvvm', _cudalib_path())
        if not os.path.exists(libdir) or not os.path.isdir(libdir):
            # If that doesn't exist either, assume we don't have the NVIDIA
            # conda package
            return

    paths = find_lib('nvvm', libdir=libdir)
    if not paths:
        return
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_nvidia_libdevice_ctk():
    """Return path to directory containing the libdevice library.
    """
    nvvm_ctk = get_nvidia_nvvm_ctk()
    if not nvvm_ctk:
        return
    nvvm_dir = os.path.dirname(nvvm_ctk)
    return os.path.join(nvvm_dir, 'libdevice')


def get_nvidia_cudalib_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit.
    """
    nvvm_ctk = get_nvidia_nvvm_ctk()
    if not nvvm_ctk:
        return
    env_dir = os.path.dirname(os.path.dirname(nvvm_ctk))
    subdir = 'bin' if IS_WIN32 else 'lib'
    return os.path.join(env_dir, subdir)


def get_nvidia_static_cudalib_ctk():
    """Return path to directory containing the static libraries of cudatoolkit.
    """
    nvvm_ctk = get_nvidia_nvvm_ctk()
    if not nvvm_ctk:
        return

    if IS_WIN32 and ("Library" not in nvvm_ctk):
        # Location specific to CUDA 11.x packages on Windows
        dirs = ('Lib', 'x64')
    else:
        # Linux, or Windows with CUDA 12.x packages
        dirs = ('lib',)

    env_dir = os.path.dirname(os.path.dirname(nvvm_ctk))
    return os.path.join(env_dir, *dirs)


def get_cuda_home(*subdirs):
    """Get paths of CUDA_HOME.
    If *subdirs* are the subdirectory name to be appended in the resulting
    path.
    """
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        # Try Windows CUDA installation without Anaconda
        cuda_home = os.environ.get('CUDA_PATH')
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)


def _get_nvvm_path():
    by, path = _get_nvvm_path_decision()
    if by == "NVIDIA NVCC Wheel":
        # The NVVM path is a directory, not a file
        path = os.path.join(path, "libnvvm.so")
    else:
        candidates = find_lib('nvvm', path)
        path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)


def _get_nvrtc_path():
    by, path = _get_nvrtc_path_decision()
    if by == "NVIDIA NVCC Wheel":
        path = str(path)
    else:
        candidates = find_lib('nvrtc', path)
        path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)


def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvvm": file_path
    - "libdevice": List[Tuple[arch, file_path]]
    - "cudalib_dir": directory_path

    Note: The result of the function is cached.
    """
    # Check cache
    if hasattr(get_cuda_paths, '_cached_result'):
        return get_cuda_paths._cached_result
    else:
        # Not in cache
        d = {
            'nvvm': _get_nvvm_path(),
            'nvrtc': _get_nvrtc_path(),
            'libdevice': _get_libdevice_paths(),
            'cudalib_dir': _get_cudalib_dir(),
            'static_cudalib_dir': _get_static_cudalib_dir(),
            'include_dir': _get_include_dir(),
        }
        # Cache result
        get_cuda_paths._cached_result = d
        return d


def get_debian_pkg_libdevice():
    """
    Return the Debian NVIDIA Maintainers-packaged libdevice location, if it
    exists.
    """
    pkg_libdevice_location = '/usr/lib/nvidia-cuda-toolkit/libdevice'
    if not os.path.exists(pkg_libdevice_location):
        return None
    return pkg_libdevice_location


def get_libdevice_wheel():
    nvvm_path = _get_nvvm_wheel()
    if nvvm_path is None:
        return None
    nvvm_path = Path(nvvm_path)
    libdevice_path = nvvm_path.parent / "libdevice"

    return str(libdevice_path)


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
        arch_to_targets = {
            'x86_64': 'x86_64-linux',
            'aarch64': 'sbsa-linux'
        }
    elif system == "Windows":
        arch_to_targets = {
            'AMD64': 'x64',
        }
    else:
        arch_to_targets = {}

    return arch_to_targets.get(machine, None)


def get_conda_include_dir():
    """
    Return the include directory in the current conda environment, if one
    is active and it exists.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return

    if platform.system() == "Windows":
        include_dir = os.path.join(
            sys.prefix, 'Library', 'include'
        )
    elif target_name := get_current_cuda_target_name():
        include_dir = os.path.join(
            sys.prefix, 'targets', target_name, 'include'
        )
    else:
        # A fallback when target cannot determined
        # though usually it shouldn't.
        include_dir = os.path.join(sys.prefix, 'include')

    if (os.path.exists(include_dir) and os.path.isdir(include_dir)
            and os.path.exists(os.path.join(include_dir,
                                            'cuda_device_runtime_api.h'))):
        return include_dir
    return


def _get_include_dir():
    """Find the root include directory."""
    options = [
        ('Conda environment (NVIDIA package)', get_conda_include_dir()),
        ('CUDA_INCLUDE_PATH Config Entry', config.CUDA_INCLUDE_PATH),
        # TODO: add others
    ]
    by, include_dir = _find_valid_path(options)
    return _env_path_tuple(by, include_dir)
