..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-cuda-installation:

============
Installation
============

Requirements
============

Supported GPUs
--------------

Numba supports all NVIDIA GPUs that are supported by the CUDA Toolkit it uses.
Presently for CUDA 12 this ranges from Compute Capabilities 5.0 to 12.1
depending on the exact installed version, and for CUDA 13 this ranges from 7.5
to 12.1 (the latest as of CUDA 13.0).


Supported CUDA Toolkits
-----------------------

Numba-CUDA aims to support all minor versions of the two most recent CUDA
Toolkit releases. Presently 12 and 13 are supported.

For further information about version compatibility between toolkit and driver
versions, refer to :ref:`minor-version-compatibility`.


Installation with a Python package manager
==========================================

Conda users can install the CUDA Toolkit into a conda environment::

    $ conda install -c conda-forge numba-cuda "cuda-version=12"

Or for CUDA 13::

    $ conda install -c conda-forge numba-cuda "cuda-version=13"

Alternatively, you can install all CUDA 12 dependencies from PyPI via ``pip``::

    $ pip install numba-cuda[cu12]

CUDA 13 dependencies can be installed via ``pip`` with::

    $ pip install numba-cuda[cu13]

If you are not using Conda/pip or if you want to use a different version of CUDA
toolkit, :ref:`cudatoolkit-lookup` describes how Numba searches for a CUDA toolkit.


Configuration
=============

.. _cuda-bindings:

CUDA Bindings
-------------

Numba-CUDA uses the `NVIDIA CUDA Python bindings <https://nvidia.github.io/cuda-python/>`_
for interacting with the CUDA Driver API. The legacy ctypes bindings and the
``NUMBA_CUDA_USE_NVIDIA_BINDING`` environment variable have been removed.


.. _cudatoolkit-lookup:

CUDA Driver and Toolkit search paths
------------------------------------

Default behavior
~~~~~~~~~~~~~~~~

Searches for the CUDA driver and toolkit libraries use the NVIDIA bindings'
`built-in path-finding logic <https://github.com/NVIDIA/cuda-python/tree/main/cuda_bindings/cuda/bindings/_path_finder>`_.

In addition to the CUDA toolkit libraries, which can be installed by conda into
an environment or installed system-wide by the `CUDA SDK installer
<https://developer.nvidia.com/cuda-downloads>`_, the CUDA target in Numba also
requires an up-to-date NVIDIA driver. Updated NVIDIA drivers are also installed
by the CUDA SDK installer, so there is no need to do both. When using the NVIDIA
bindings (default), driver discovery is handled by their path-finding logic, and
no Numba-specific environment variables are needed to locate ``libcuda``.
