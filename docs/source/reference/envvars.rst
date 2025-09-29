..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-envvars-gpu-support:

Environment Variables
---------------------

Various environment variables can be set to affect the behavior of the CUDA
target.

.. envvar:: NUMBA_DISABLE_CUDA

   If set to non-zero, disable CUDA support.

.. envvar:: NUMBA_FORCE_CUDA_CC

   If set, force the CUDA compute capability to the given version (a
   string of the type ``major.minor``), regardless of attached devices.

.. envvar:: NUMBA_CUDA_DEFAULT_PTX_CC

   The default compute capability (a string of the type ``major.minor``) to
   target when compiling to PTX using ``cuda.compile_ptx``. The default is
   5.0, which is the lowest non-deprecated compute capability in the most
   recent version of the CUDA toolkit supported (12.4 at present).

.. envvar:: NUMBA_ENABLE_CUDASIM

   If set, don't compile and execute code for the GPU, but use the CUDA
   Simulator instead. For debugging purposes.


.. envvar:: NUMBA_CUDA_ARRAY_INTERFACE_SYNC

   Whether to synchronize on streams provided by objects imported using the CUDA
   Array Interface. This defaults to 1. If set to 0, then no synchronization
   takes place, and the user of Numba (and other CUDA libraries) is responsible
   for ensuring correctness with respect to synchronization on streams.

.. envvar:: NUMBA_CUDA_LOG_LEVEL

   For debugging purposes. If no other logging is configured, the value of this
   variable is the logging level for CUDA API calls. The default value is
   ``CRITICAL`` - to trace all API calls on standard error, set this to
   ``DEBUG``.

.. envvar:: NUMBA_CUDA_LOG_API_ARGS

   By default the CUDA API call logs only give the names of functions called.
   Setting this variable to 1 also includes the values of arguments to Driver
   API calls in the logs.



.. envvar:: NUMBA_CUDA_LOG_SIZE

   Buffer size for logs produced by CUDA driver API operations. This defaults
   to 1024 and should not normally need to be modified - however, if an error
   in an API call produces a large amount of output that appears to be
   truncated (perhaps due to multiple long function names, for example) then
   this variable can be used to increase the buffer size and view the full
   error message.

.. envvar:: NUMBA_CUDA_VERBOSE_JIT_LOG

   Whether the CUDA driver should produce verbose log messages. Defaults to 1,
   indicating that verbose messaging is enabled. This should not need to be
   modified under normal circumstances.

.. envvar:: NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM

   When set to 1, the default stream is the per-thread default stream. When set
   to 0, the default stream is the legacy default stream. This defaults to 0,
   for the legacy default stream. See `Stream Synchronization Behavior
   <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_
   for an explanation of the legacy and per-thread default streams.

   This variable only takes effect when using Numba's internal CUDA bindings;
   when using the NVIDIA bindings, use the environment variable
   ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` instead.

   .. seealso::

      The `Runtime Environment Variables section
      <https://nvidia.github.io/cuda-python/cuda-bindings/latest/environment_variables.html#runtime-environment-variables>`_
      in the NVIDIA Bindings documentation.

.. envvar:: NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS

   Enable warnings if the grid size is too small relative to the number of
   streaming multiprocessors (SM). This option is on by default (default value is 1).

   The heuristic checked is whether ``gridsize < 2 * (number of SMs)``. NOTE: The absence of
   a warning does not imply a good gridsize relative to the number of SMs. Disabling
   this warning will reduce the number of CUDA API calls (during JIT compilation), as the
   heuristic needs to check the number of SMs available on the device in the
   current context.

.. envvar:: NUMBA_CUDA_WARN_ON_IMPLICIT_COPY

   Enable warnings if a kernel is launched with host memory which forces a copy to and
   from the device. This option is on by default (default value is 1).

.. envvar:: NUMBA_CUDA_USE_NVIDIA_BINDING

   When set to 1, Numba will attempt to use the `NVIDIA CUDA Python binding
   <https://nvidia.github.io/cuda-python/>`_ to make calls to the driver API
   instead of using its own ctypes binding. This defaults to 1 (on). Set to
   0 to use the ctypes bindings.

.. envvar:: NUMBA_CUDA_INCLUDE_PATH

   The location of the CUDA include files. This is used when linking CUDA C/C++
   sources to Python kernels, and needs to be correctly set for CUDA includes to
   be available to linked C/C++ sources. On Linux, it defaults to
   ``/usr/local/cuda/include``. On Windows, the default is
   ``$env:CUDA_PATH\include``.

.. envvar:: NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS

   A colon separated list of paths that Numba's NVRTC should search for when compiling
   external functions. These folders are searched after the system cudatoolkit search
   paths and Numba-CUDA's internal search paths.
