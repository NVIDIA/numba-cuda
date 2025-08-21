# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from warnings import warn
from numba.core import types, config
from numba.core.errors import DeprecationError, NumbaInvalidConfigWarning
from numba.cuda.compiler import declare_device_function
from numba.cuda.core import sigutils
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.simulator.kernel import FakeCUDAKernel
from numba.cuda.cudadrv.driver import _have_nvjitlink


_msg_deprecated_signature_arg = (
    "Deprecated keyword argument `{0}`. "
    "Signatures should be passed as the first "
    "positional argument."
)


def jit(
    func_or_sig=None,
    device=False,
    inline="never",
    forceinline=False,
    link=[],
    debug=None,
    opt=None,
    lineinfo=False,
    cache=False,
    launch_bounds=None,
    lto=None,
    **kws,
):
    """
    JIT compile a Python function for CUDA GPUs.

    :param func_or_sig: A function to JIT compile, or *signatures* of a
       function to compile. If a function is supplied, then a
       :class:`Dispatcher <numba.cuda.dispatcher.CUDADispatcher>` is returned.
       Otherwise, ``func_or_sig`` may be a signature or a list of signatures,
       and a function is returned. The returned function accepts another
       function, which it will compile and then return a :class:`Dispatcher
       <numba.cuda.dispatcher.CUDADispatcher>`. See :ref:`jit-decorator` for
       more information about passing signatures.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param inline: Enables inlining at the Numba IR level when set to
       ``"always"``. See `Notes on Inlining
       <https://numba.readthedocs.io/en/stable/developer/inlining.html>`_.
    :type inline: str
    :param forceinline: Enables inlining at the NVVM IR level when set to
       ``True``. This is accomplished by adding the ``alwaysinline`` function
       attribute to the function definition.
    :type forceinline: bool
    :param link: A list of files containing PTX or CUDA C/C++ source to link
       with the function
    :type link: list
    :param debug: If True, check for exceptions thrown when executing the
       kernel. Since this degrades performance, this should only be used for
       debugging purposes. If set to True, then ``opt`` should be set to False.
       Defaults to False.  (The default value can be overridden by setting
       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param fastmath: When True, enables fastmath optimizations as outlined in
       the :ref:`CUDA Fast Math documentation <cuda-fast-math>`.
    :param max_registers: Request that the kernel is limited to using at most
       this number of registers per thread. The limit may not be respected if
       the ABI requires a greater number of registers than that requested.
       Useful for increasing occupancy.
    :param opt: Whether to compile with optimization enabled. If unspecified,
       the OPT configuration variable is decided by ``NUMBA_OPT```; all
       non-zero values will enable optimization.
    :type opt: bool
    :param lineinfo: If True, generate a line mapping between source code and
       assembly code. This enables inspection of the source code in NVIDIA
       profiling tools and correlation with program counter sampling.
    :type lineinfo: bool
    :param cache: If True, enables the file-based cache for this function.
    :type cache: bool
    :param launch_bounds: Kernel launch bounds, specified as a scalar or a tuple
                          of between one and three items. Tuple items provide:

                          - The maximum number of threads per block,
                          - The minimum number of blocks per SM,
                          - The maximum number of blocks per cluster.

                          If a scalar is provided, it is used as the maximum
                          number of threads per block.
    :type launch_bounds: int | tuple[int]
    :param lto: Whether to enable LTO. If unspecified, LTO is enabled by
                default when nvjitlink is available, except for kernels where
                ``debug=True``.
    :type lto: bool
    """

    if link and config.ENABLE_CUDASIM:
        raise NotImplementedError("Cannot link PTX in the simulator")

    if kws.get("boundscheck"):
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if kws.get("argtypes") is not None:
        msg = _msg_deprecated_signature_arg.format("argtypes")
        raise DeprecationError(msg)
    if kws.get("restype") is not None:
        msg = _msg_deprecated_signature_arg.format("restype")
        raise DeprecationError(msg)
    if kws.get("bind") is not None:
        msg = _msg_deprecated_signature_arg.format("bind")
        raise DeprecationError(msg)

    if isinstance(inline, bool):
        DeprecationWarning(
            "Passing bool to inline argument is deprecated, please refer to "
            "Numba's documentation on inlining: "
            "https://numba.readthedocs.io/en/stable/developer/inlining.html. "
            "You may have wanted the forceinline argument instead, to force "
            "inlining at the NVVM IR level."
        )

        inline = "always" if inline else "never"

    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug
    opt = (config.OPT != 0) if opt is None else opt
    fastmath = kws.get("fastmath", False)
    extensions = kws.get("extensions", [])

    if debug and opt:
        msg = (
            "debug=True with opt=True "
            "is not supported by CUDA. This may result in a crash"
            " - set debug=False or opt=False."
        )
        warn(NumbaInvalidConfigWarning(msg))

    if debug and lineinfo:
        msg = (
            "debug and lineinfo are mutually exclusive. Use debug to get "
            "full debug info (this disables some optimizations), or "
            "lineinfo for line info only with code generation unaffected."
        )
        warn(NumbaInvalidConfigWarning(msg))

    if device and kws.get("link"):
        raise ValueError("link keyword invalid for device function")

    if lto is None:
        # Default to using LTO if nvjitlink is available and we're not debugging
        lto = _have_nvjitlink() and not debug
    else:
        if lto and not _have_nvjitlink():
            raise RuntimeError(
                "LTO requires nvjitlink, which is not available"
                "or not sufficiently recent (>=12.3)"
            )

    if sigutils.is_signature(func_or_sig):
        signatures = [func_or_sig]
        specialized = True
    elif isinstance(func_or_sig, list):
        signatures = func_or_sig
        specialized = False
    else:
        signatures = None

    if signatures is not None:
        if config.ENABLE_CUDASIM:

            def jitwrapper(func):
                return FakeCUDAKernel(func, device=device, fastmath=fastmath)

            return jitwrapper

        def _jit(func):
            targetoptions = kws.copy()
            targetoptions["debug"] = debug
            targetoptions["lineinfo"] = lineinfo
            targetoptions["link"] = link
            targetoptions["opt"] = opt
            targetoptions["fastmath"] = fastmath
            targetoptions["device"] = device
            targetoptions["inline"] = inline
            targetoptions["forceinline"] = forceinline
            targetoptions["extensions"] = extensions
            targetoptions["launch_bounds"] = launch_bounds
            targetoptions["lto"] = lto

            disp = CUDADispatcher(func, targetoptions=targetoptions)

            if cache:
                disp.enable_caching()

            for sig in signatures:
                argtypes, restype = sigutils.normalize_signature(sig)

                if restype and not device and restype != types.void:
                    raise TypeError("CUDA kernel must have void return type.")

                if device:
                    from numba.core import typeinfer

                    with typeinfer.register_dispatcher(disp):
                        disp.compile_device(argtypes, restype)
                else:
                    disp.compile(argtypes)

            disp._specialized = specialized
            disp.disable_compile()

            return disp

        return _jit
    else:
        if func_or_sig is None:
            if config.ENABLE_CUDASIM:

                def autojitwrapper(func):
                    return FakeCUDAKernel(
                        func, device=device, fastmath=fastmath
                    )
            else:

                def autojitwrapper(func):
                    return jit(
                        func,
                        device=device,
                        inline=inline,
                        forceinline=forceinline,
                        debug=debug,
                        opt=opt,
                        lineinfo=lineinfo,
                        link=link,
                        cache=cache,
                        launch_bounds=launch_bounds,
                        **kws,
                    )

            return autojitwrapper
        # func_or_sig is a function
        else:
            if config.ENABLE_CUDASIM:
                return FakeCUDAKernel(
                    func_or_sig, device=device, fastmath=fastmath
                )
            else:
                targetoptions = kws.copy()
                targetoptions["debug"] = debug
                targetoptions["lineinfo"] = lineinfo
                targetoptions["opt"] = opt
                targetoptions["link"] = link
                targetoptions["fastmath"] = fastmath
                targetoptions["device"] = device
                targetoptions["inline"] = inline
                targetoptions["forceinline"] = forceinline
                targetoptions["extensions"] = extensions
                targetoptions["launch_bounds"] = launch_bounds
                targetoptions["lto"] = lto
                disp = CUDADispatcher(func_or_sig, targetoptions=targetoptions)

                if cache:
                    disp.enable_caching()

                return disp


def declare_device(name, sig, link=None, use_cooperative=False):
    """
    Declare the signature of a foreign function. Returns a descriptor that can
    be used to call the function from a Python kernel.

    :param name: The name of the foreign function.
    :type name: str
    :param sig: The Numba signature of the function.
    :param link: External code to link when calling the function.
    :param use_cooperative: External code requires cooperative launch.
    """
    if link is None:
        link = tuple()
    else:
        if not isinstance(link, (list, tuple, set)):
            link = (link,)

    argtypes, restype = sigutils.normalize_signature(sig)
    if restype is None:
        msg = "Return type must be provided for device declarations"
        raise TypeError(msg)

    template = declare_device_function(
        name, restype, argtypes, link, use_cooperative
    )

    return template.key
