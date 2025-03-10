from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER, CDLL, RTLD_GLOBAL
from enum import IntEnum
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
                                      NvrtcSupportError)
from numba.cuda.cuda_paths import get_cuda_paths
import functools
import os
import re
import site
import sys
import threading
import warnings

# Opaque handle for compilation unit
nvrtc_program = c_void_p

# Result code
nvrtc_result = c_int

PLATFORM_LINUX = sys.platform.startswith("linux")
PLATFORM_WIN = sys.platform.startswith("win32")


class NvrtcResult(IntEnum):
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_OUT_OF_MEMORY = 1
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    NVRTC_ERROR_INVALID_INPUT = 3
    NVRTC_ERROR_INVALID_PROGRAM = 4
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
    NVRTC_ERROR_INTERNAL_ERROR = 11


_nvrtc_lock = threading.Lock()
_nvrtc_obj = []

def force_loading_nvrtc(cu_ver):
    # this logic should live in CUDA Python...
    # TODO: remove this function once NVIDIA/cuda-python#62 is resolved
    # This logic handles all cases - wheel, conda, and system installations
    global _nvrtc_obj
    if len(_nvrtc_obj) > 0:
        return

    cu_ver = cu_ver.split(".")
    major = cu_ver[0]
    if major == "11":
        # CUDA 11.2+ supports minor ver compat
        if PLATFORM_LINUX:
            cu_ver = "11.2"
        elif PLATFORM_WIN:
            cu_ver = "112"
    elif major == "12":
        if PLATFORM_LINUX:
            cu_ver = "12"
        elif PLATFORM_WIN:
            cu_ver = "120"
    else:
        raise NotImplementedError(f"CUDA {major} is not supported")

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + [None]
    for sp in site_paths:
        if PLATFORM_LINUX:
            dso_dir = "lib"
            dso_path = f"libnvrtc.so.{cu_ver}"
        elif PLATFORM_WIN:
            dso_dir = "bin"
            dso_path = f"nvrtc64_{cu_ver}_0.dll"
        else:
            raise AssertionError()

        if sp is not None:
            dso_dir = os.path.join(sp, "nvidia", "cuda_nvrtc", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _nvrtc_obj.append(CDLL(dso_path, mode=RTLD_GLOBAL))
        except OSError:
            continue
        else:
            if PLATFORM_WIN:
                import win32api

                # This absolute path will always be correct regardless of the package source
                nvrtc_path = win32api.GetModuleFileNameW(_nvrtc_obj[0]._handle)
                dso_dir = os.path.dirname(nvrtc_path)
                dso_path = os.path.join(dso_dir, [f for f in os.listdir(dso_dir) if re.match("^nvrtc-builtins.*.dll$", f)][0])
                _nvrtc_obj.append(CDLL(dso_path))
            break
    else:
        raise RuntimeError(
            f"NVRTC from CUDA {major} not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            f"you may need to perform one of the steps below:\n"
            f"  - pip install nvidia-cuda-nvrtc-cu{major}\n"
            f"  - conda install -c conda-forge cuda-nvrtc cuda-version={major}\n"
            "  - export LD_LIBRARY_PATH=/path/to/CUDA/Toolkit/lib64:$LD_LIBRARY_PATH"
        )

class NvrtcProgram:
    """
    A class for managing the lifetime of nvrtcProgram instances. Instances of
    the class own an nvrtcProgram; when an instance is deleted, the underlying
    nvrtcProgram is destroyed using the appropriate NVRTC API.
    """
    def __init__(self, nvrtc, handle):
        self._nvrtc = nvrtc
        self._handle = handle

    @property
    def handle(self):
        return self._handle

    def __del__(self):
        if self._handle:
            self._nvrtc.destroy_program(self)


class NVRTC:
    """
    Provides a Pythonic interface to the NVRTC APIs, abstracting away the C API
    calls.

    The sole instance of this class is a process-wide singleton, similar to the
    NVVM interface. Initialization is protected by a lock and uses the standard
    (for Numba) open_cudalib function to load the NVRTC library.
    """

    _CU12ONLY_PROTOTYPES = {
        # nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *ltoSizeRet);
        "nvrtcGetLTOIRSize": (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *lto);
        "nvrtcGetLTOIR": (nvrtc_result, nvrtc_program, c_char_p)
    }

    _PROTOTYPES = {
        # nvrtcResult nvrtcVersion(int *major, int *minor)
        'nvrtcVersion': (nvrtc_result, POINTER(c_int), POINTER(c_int)),
        # nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
        #                                const char *src,
        #                                const char *name,
        #                                int numHeaders,
        #                                const char * const *headers,
        #                                const char * const *includeNames)
        'nvrtcCreateProgram': (nvrtc_result, nvrtc_program, c_char_p, c_char_p,
                               c_int, POINTER(c_char_p), POINTER(c_char_p)),
        # nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
        'nvrtcDestroyProgram': (nvrtc_result, POINTER(nvrtc_program)),
        # nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
        #                                 int numOptions,
        #                                 const char * const *options)
        'nvrtcCompileProgram': (nvrtc_result, nvrtc_program, c_int,
                                POINTER(c_char_p)),
        # nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);
        'nvrtcGetPTXSize': (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
        'nvrtcGetPTX': (nvrtc_result, nvrtc_program, c_char_p),
        # nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog,
        #                               size_t *cubinSizeRet);
        'nvrtcGetCUBINSize': (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin);
        'nvrtcGetCUBIN': (nvrtc_result, nvrtc_program, c_char_p),
        # nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog,
        #                                    size_t *logSizeRet);
        'nvrtcGetProgramLogSize': (nvrtc_result, nvrtc_program,
                                   POINTER(c_size_t)),
        # nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);
        'nvrtcGetProgramLog': (nvrtc_result, nvrtc_program, c_char_p),
    }

    # Singleton reference
    __INSTANCE = None

    def __new__(cls):
        with _nvrtc_lock:
            if cls.__INSTANCE is None:
                cls.__INSTANCE = inst = object.__new__(cls)
                try:
                    lib = _nvrtc_obj[0]
                except OSError as e:
                    cls.__INSTANCE = None
                    raise NvrtcSupportError("NVRTC cannot be loaded") from e

                from numba.cuda.cudadrv.runtime import get_version
                if get_version() >= (12, 0):
                    inst._PROTOTYPES |= inst._CU12ONLY_PROTOTYPES

                # Find & populate functions
                for name, proto in inst._PROTOTYPES.items():
                    func = getattr(lib, name)
                    func.restype = proto[0]
                    func.argtypes = proto[1:]

                    @functools.wraps(func)
                    def checked_call(*args, func=func, name=name):
                        error = func(*args)
                        if error == NvrtcResult.NVRTC_ERROR_COMPILATION:
                            raise NvrtcCompilationError()
                        elif error != NvrtcResult.NVRTC_SUCCESS:
                            try:
                                error_name = NvrtcResult(error).name
                            except ValueError:
                                error_name = ('Unknown nvrtc_result '
                                              f'(error code: {error})')
                            msg = f'Failed to call {name}: {error_name}'
                            raise NvrtcError(msg)

                    setattr(inst, name, checked_call)

        return cls.__INSTANCE


    def get_version(self):
        """
        Get the NVRTC version as a tuple (major, minor).
        """
        major = c_int()
        minor = c_int()
        self.nvrtcVersion(byref(major), byref(minor))
        return major.value, minor.value

    def create_program(self, src, name):
        """
        Create an NVRTC program with managed lifetime.
        """
        if isinstance(src, str):
            src = src.encode()
        if isinstance(name, str):
            name = name.encode()

        handle = nvrtc_program()

        # The final three arguments are for passing the contents of headers -
        # this is not supported, so there are 0 headers and the header names
        # and contents are null.
        self.nvrtcCreateProgram(byref(handle), src, name, 0, None, None)
        return NvrtcProgram(self, handle)

    def compile_program(self, program, options):
        """
        Compile an NVRTC program. Compilation may fail due to a user error in
        the source; this function returns ``True`` if there is a compilation
        error and ``False`` on success.
        """
        # We hold a list of encoded options to ensure they can't be collected
        # prior to the call to nvrtcCompileProgram
        encoded_options = [opt.encode() for opt in options]
        option_pointers = [c_char_p(opt) for opt in encoded_options]
        c_options_type = (c_char_p * len(options))
        c_options = c_options_type(*option_pointers)
        try:
            self.nvrtcCompileProgram(program.handle, len(options), c_options)
            return False
        except NvrtcCompilationError:
            return True

    def destroy_program(self, program):
        """
        Destroy an NVRTC program.
        """
        self.nvrtcDestroyProgram(byref(program.handle))

    def get_compile_log(self, program):
        """
        Get the compile log as a Python string.
        """
        log_size = c_size_t()
        self.nvrtcGetProgramLogSize(program.handle, byref(log_size))

        log = (c_char * log_size.value)()
        self.nvrtcGetProgramLog(program.handle, log)

        return log.value.decode()

    def get_ptx(self, program):
        """
        Get the compiled PTX as a Python string.
        """
        ptx_size = c_size_t()
        self.nvrtcGetPTXSize(program.handle, byref(ptx_size))

        ptx = (c_char * ptx_size.value)()
        self.nvrtcGetPTX(program.handle, ptx)

        return ptx.value.decode()

    def get_lto(self, program):
        """
        Get the compiled LTOIR as a Python bytes object.
        """
        lto_size = c_size_t()
        self.nvrtcGetLTOIRSize(program.handle, byref(lto_size))

        lto = b" " * lto_size.value
        self.nvrtcGetLTOIR(program.handle, lto)

        return lto


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
    nvrtc = NVRTC()
    program = nvrtc.create_program(src, name)

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (rdc) is needed to prevent device functions
    #   being optimized away.
    major, minor = cc
    arch = f'--gpu-architecture=compute_{major}{minor}'

    cuda_include = [
        f"-I{get_cuda_paths()['include_dir'].info}",
    ]

    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
    numba_cuda_path = os.path.dirname(cudadrv_path)
    numba_include = f'-I{numba_cuda_path}'

    nrt_path = os.path.join(numba_cuda_path, "runtime")
    nrt_include = f'-I{nrt_path}'

    options = [arch, *cuda_include, numba_include, nrt_include, '-rdc', 'true']

    if ltoir:
        options.append("-dlto")

    if nvrtc.get_version() < (12, 0):
        options += ["-std=c++17"]

    # Compile the program
    compile_error = nvrtc.compile_program(program, options)

    # Get log from compilation
    log = nvrtc.get_compile_log(program)

    # If the compile failed, provide the log in an exception
    if compile_error:
        msg = (f'NVRTC Compilation failure whilst compiling {name}:\n\n{log}')
        raise NvrtcError(msg)

    # Otherwise, if there's any content in the log, present it as a warning
    if log:
        msg = (f"NVRTC log messages whilst compiling {name}:\n\n{log}")
        warnings.warn(msg)

    if ltoir:
        ltoir = nvrtc.get_lto(program)
        return ltoir, log
    else:
        ptx = nvrtc.get_ptx(program)
        return ptx, log

force_loading_nvrtc('12')
