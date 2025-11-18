..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _minor-version-compatibility:

CUDA Minor Version Compatibility
================================

CUDA `Minor Version Compatibility
<https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_
(MVC) enables the use of a newer CUDA Toolkit version than the CUDA version
supported by the driver, provided that the Toolkit and driver both have the same
major version. For example, use of CUDA Toolkit 12.9 with CUDA driver 570 (CUDA
version 12.8) is supported through MVC.

Numba supports MVC using the linker in the NVIDIA CUDA Python bindings, which
uses ``nvjitlink`` to provide MVC.


References
----------

Further information about Minor Version Compatibility may be found in:

- The `CUDA Compatibility Guide
  <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>`_.
