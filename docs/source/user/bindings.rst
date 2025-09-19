..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

CUDA Bindings
=============

Numba-CUDA uses the official `NVIDIA CUDA Python bindings
<https://nvidia.github.io/cuda-python/>`_ for all CUDA Driver interactions.
The legacy internal ctypes bindings have been removed and the
``NUMBA_CUDA_USE_NVIDIA_BINDING`` environment variable no longer has any effect.


Per-Thread Default Streams
--------------------------

Responsibility for handling Per-Thread Default Streams (PTDS) is delegated to
the NVIDIA bindings. To use PTDS, set the environment variable
``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` to ``1`` instead of Numba's
environment variable :envvar:`NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM`.

.. seealso::

   The `Default Stream section
   <https://nvidia.github.io/cuda-python/release/11.6.0-notes.html#default-stream>`_
   in the NVIDIA Bindings documentation.


Roadmap
-------

The ctypes-based internal bindings have been removed in favor of the NVIDIA
bindings. Future work focuses on expanding usage of ``cuda.core`` APIs.
