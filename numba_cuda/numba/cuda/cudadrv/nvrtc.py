from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.cuda.cudadrv.error import (
    CCSupportError,
    NvrtcError,
    NvrtcBuiltinOperationFailure,
    NvrtcCompilationError,
    NvrtcSupportError,
)
from numba import config
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.utils import _readenv

import functools
import os
import threading
import warnings

NVRTC_EXTRA_SEARCH_PATHS = _readenv(
    "NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS", str, ""
) or getattr(config, "NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS", "")
if not hasattr(config, "NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS"):
    config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = NVRTC_EXTRA_SEARCH_PATHS

# Opaque handle for compilation unit
nvrtc_program = c_void_p

# Result code
nvrtc_result = c_int

if config.CUDA_USE_NVIDIA_BINDING:
    from cuda.core.experimental import Program, ProgramOptions


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

    _CU11_2ONLY_PROTOTYPES = {
        # nvrtcResult nvrtcGetNumSupportedArchs(int *numArchs);
        "nvrtcGetNumSupportedArchs": (nvrtc_result, POINTER(c_int)),
        # nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs);
        "nvrtcGetSupportedArchs": (nvrtc_result, POINTER(c_int)),
    }

    _CU12ONLY_PROTOTYPES = {
        # nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *ltoSizeRet);
        "nvrtcGetLTOIRSize": (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *lto);
        "nvrtcGetLTOIR": (nvrtc_result, nvrtc_program, c_char_p),
    }

    _PROTOTYPES = {
        # nvrtcResult nvrtcVersion(int *major, int *minor)
        "nvrtcVersion": (nvrtc_result, POINTER(c_int), POINTER(c_int)),
        # nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
        #                                const char *src,
        #                                const char *name,
        #                                int numHeaders,
        #                                const char * const *headers,
        #                                const char * const *includeNames)
        "nvrtcCreateProgram": (
            nvrtc_result,
            nvrtc_program,
            c_char_p,
            c_char_p,
            c_int,
            POINTER(c_char_p),
            POINTER(c_char_p),
        ),
        # nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
        "nvrtcDestroyProgram": (nvrtc_result, POINTER(nvrtc_program)),
        # nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
        #                                 int numOptions,
        #                                 const char * const *options)
        "nvrtcCompileProgram": (
            nvrtc_result,
            nvrtc_program,
            c_int,
            POINTER(c_char_p),
        ),
        # nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);
        "nvrtcGetPTXSize": (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
        "nvrtcGetPTX": (nvrtc_result, nvrtc_program, c_char_p),
        # nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog,
        #                               size_t *cubinSizeRet);
        "nvrtcGetCUBINSize": (nvrtc_result, nvrtc_program, POINTER(c_size_t)),
        # nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin);
        "nvrtcGetCUBIN": (nvrtc_result, nvrtc_program, c_char_p),
        # nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog,
        #                                    size_t *logSizeRet);
        "nvrtcGetProgramLogSize": (
            nvrtc_result,
            nvrtc_program,
            POINTER(c_size_t),
        ),
        # nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);
        "nvrtcGetProgramLog": (nvrtc_result, nvrtc_program, c_char_p),
    }

    # Singleton reference
    __INSTANCE = None

    def __new__(cls):
        with _nvrtc_lock:
            if cls.__INSTANCE is None:
                from numba.cuda.cudadrv.libs import open_cudalib

                cls.__INSTANCE = inst = object.__new__(cls)
                try:
                    lib = open_cudalib("nvrtc")
                except OSError as e:
                    cls.__INSTANCE = None
                    raise NvrtcSupportError("NVRTC cannot be loaded") from e

                from numba.cuda.cudadrv.runtime import get_version

                if get_version() >= (11, 2):
                    inst._PROTOTYPES |= inst._CU11_2ONLY_PROTOTYPES
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
                        elif (
                            error
                            == NvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
                        ):
                            raise NvrtcBuiltinOperationFailure()
                        elif error != NvrtcResult.NVRTC_SUCCESS:
                            try:
                                error_name = NvrtcResult(error).name
                            except ValueError:
                                error_name = (
                                    "Unknown nvrtc_result "
                                    f"(error code: {error})"
                                )
                            msg = f"Failed to call {name}: {error_name}"
                            raise NvrtcError(msg)

                    setattr(inst, name, checked_call)

        return cls.__INSTANCE

    @functools.cache
    def get_supported_archs(self):
        """
        Get Supported Architectures by NVRTC as list of arch tuples.
        """
        num = c_int()
        self.nvrtcGetNumSupportedArchs(byref(num))
        archs = (c_int * num.value)()
        self.nvrtcGetSupportedArchs(archs)
        return [(archs[i] // 10, archs[i] % 10) for i in range(num.value)]

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
        c_options_type = c_char_p * len(options)
        c_options = c_options_type(*option_pointers)
        try:
            self.nvrtcCompileProgram(program.handle, len(options), c_options)
            return False
        except (NvrtcCompilationError, NvrtcBuiltinOperationFailure):
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

    version = nvrtc.get_version()
    ver_str = lambda v: ".".join(v)
    if version < (11, 2):
        raise RuntimeError(
            "Unsupported CUDA version. CUDA 11.2 or higher is required."
        )
    else:
        supported_arch = nvrtc.get_supported_archs()
        try:
            found = max(filter(lambda v: v <= cc, [v for v in supported_arch]))
        except ValueError:
            raise RuntimeError(
                f"Device compute capability {ver_str(cc)} is less than the "
                f"minimum supported by NVRTC {ver_str(version)}. Supported "
                "compute capabilities are "
                f"{', '.join([ver_str(v) for v in supported_arch])}."
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

    if config.CUDA_USE_NVIDIA_BINDING:
        arch = f"sm_{major}{minor}"
    else:
        arch = f"--gpu-architecture=compute_{major}{minor}"

    cuda_include = [
        f"{get_cuda_paths()['include_dir'].info}",
    ]

    nvrtc_version = nvrtc.get_version()
    nvrtc_ver_major = nvrtc_version[0]

    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
    numba_cuda_path = os.path.dirname(cudadrv_path)

    if nvrtc_ver_major == 11:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '11')}"
    else:
        numba_include = f"{os.path.join(numba_cuda_path, 'include', '12')}"

    if config.CUDA_NVRTC_EXTRA_SEARCH_PATHS:
        extra_includes = config.CUDA_NVRTC_EXTRA_SEARCH_PATHS.split(":")
    else:
        extra_includes = []

    nrt_include = os.path.join(numba_cuda_path, "memory_management")

    includes = [numba_include, *cuda_include, nrt_include, *extra_includes]

    if config.CUDA_USE_NVIDIA_BINDING:
        options = ProgramOptions(
            arch=arch,
            include_path=includes,
            relocatable_device_code=True,
            std="c++17" if nvrtc_version < (12, 0) else None,
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

    else:
        includes = [f"-I{path}" for path in includes]
        options = [
            arch,
            *includes,
            "-rdc",
            "true",
        ]

        if ltoir:
            options.append("-dlto")

        if nvrtc_version < (12, 0):
            options.append("-std=c++17")

        # Compile the program
        compile_error = nvrtc.compile_program(program, options)

        # Get log from compilation
        log = nvrtc.get_compile_log(program)

        # If the compile failed, provide the log in an exception
        if compile_error:
            msg = f"NVRTC Compilation failure whilst compiling {name}:\n\n{log}"
            raise NvrtcError(msg)

        # Otherwise, if there's any content in the log, present it as a warning
        if log:
            msg = f"NVRTC log messages whilst compiling {name}:\n\n{log}"
            warnings.warn(msg)

        if ltoir:
            ltoir = nvrtc.get_lto(program)
            return ltoir, log
        else:
            ptx = nvrtc.get_ptx(program)
            return ptx, log


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
    return NVRTC().get_supported_archs()
