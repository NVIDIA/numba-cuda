..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. Numba CUDA documentation master file, created by
   sphinx-quickstart on Wed Jun 12 13:41:17 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Numba-CUDA
==========

Numba-CUDA provides a CUDA target for the Numba Python JIT Compiler. It is used
for writing SIMT kernels in Python, for providing Python bindings for
accelerated device libraries, and as a compiler for user-defined functions in
accelerated libraries like `RAPIDS <https://rapids.ai>`_.

* To install Numba-CUDA, see: :ref:`numba-cuda-installation`.
* To get started writing CUDA kernels in Python with Numba, see
  :ref:`writing-cuda-kernels`.
* Browse the :ref:`numba-cuda-examples` to see a variety of use cases of Numba-CUDA.

Maintenance Notice
==================

.. note::

   Numba-CUDA is in maintenance mode. Moving forward, we intend to support only
   security issues and critical bug fixes through the lifetime of CUDA 13. New
   feature development is targeted towards `Numba-CUDA-MLIR
   <https://nvidia.github.io/numba-cuda-mlir>`_, and we recommend upgrading to
   Numba-CUDA-MLIR as soon as practical.

   For migration guidance, see `Migration from Numba / Numba-CUDA
   <https://github.com/NVIDIA/numba-cuda-mlir#migration-from-numba--numba-cuda>`_.

Contents
========

.. toctree::
   :maxdepth: 2

   user/index.rst
   reference/index.rst
