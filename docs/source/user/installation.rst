.. _numba-cuda-installation:

============
Installation
============

Requirements
============

Supported GPUs
--------------

Numba supports all NVIDIA GPUs that are supported by the CUDA Toolkit it uses.
Presently for CUDA 11 this ranges from Compute Capabilities 3.5 to 9.0, and for
CUDA 12 this ranges from 5.0 to 12.1, depending on the exact installed version.


Supported CUDA Toolkits
-----------------------

Numba-CUDA aims to support all minor versions of the two most recent CUDA
Toolkit releases. Presently 11 and 12 are supported; CUDA 11.2 is the minimum
required, because older releases (11.0 and 11.1) have a version of NVVM based on
a previous and incompatible LLVM version.

For further information about version compatibility between toolkit and driver
versions, refer to :ref:`minor-version-compatibility`.


Installation with a Python package manager
==========================================

Conda users can install the CUDA Toolkit into a conda environment.

For CUDA 12, ``cuda-nvcc`` and ``cuda-nvrtc`` are required::

    $ conda install -c conda-forge numba-cuda cuda-nvcc cuda-nvrtc "cuda-version>=12.0"

Alternatively, you can install all CUDA 12 dependencies from PyPI via ``pip``::

    $ pip install numba-cuda[cu12]

For CUDA 11, ``cudatoolkit`` is required::

    $ conda install -c conda-forge numba-cuda cudatoolkit "cuda-version>=11.2,<12.0"

or::

    $ pip install numba-cuda[cu11]

If you are not using Conda/pip or if you want to use a different version of CUDA
toolkit, :ref:`cudatoolkit-lookup` describes how Numba searches for a CUDA toolkit.


Configuration
=============

.. _cuda-bindings:

CUDA Bindings
-------------

Numba supports interacting with the CUDA Driver API via either the `NVIDIA CUDA
Python bindings <https://nvidia.github.io/cuda-python/>`_ or its own ctypes-based
bindings. Functionality is equivalent between the two binding choices. The
NVIDIA bindings are the default, and the ctypes bindings are now deprecated.

If the NVIDIA bindings are not present in your environment, you can install them
with::

   $ conda install -c conda-forge cuda-bindings

if you are using Conda, or::

   $ pip install cuda-bindings[cu11]

for CUDA 11 bindings with pip, or::

   $ pip install cuda-bindings[cu12]

for CUDA 12 bindings with pip. Note that the bracket notation
``numba-cuda[cuXX]`` introduced above will bring in this dependency for you.

If you do not want to use the NVIDIA bindings, the (deprecated) ctypes bindings
can be enabled by setting the environment variable
:envvar:`NUMBA_CUDA_USE_NVIDIA_BINDING` to ``"0"``.


.. _cudatoolkit-lookup:

Setting CUDA Installation Path
------------------------------

Numba searches for a CUDA toolkit installation in the following order:

1. Conda-installed CUDA Toolkit packages
2. Pip-installed CUDA Toolkit packages
3. The environment variable ``CUDA_HOME``, which points to the directory of the
   installed CUDA toolkit (i.e. ``/home/user/cuda-12``)
4. System-wide installation at exactly ``/usr/local/cuda`` on Linux platforms.
   Versioned installation paths (i.e. ``/usr/local/cuda-12.0``) are intentionally
   ignored. Users can use ``CUDA_HOME`` to select specific versions.

In addition to the CUDA toolkit libraries, which can be installed by conda into
an environment or installed system-wide by the `CUDA SDK installer
<https://developer.nvidia.com/cuda-downloads>`_, the CUDA target in Numba
also requires an up-to-date NVIDIA graphics driver.  Updated graphics drivers
are also installed by the CUDA SDK installer, so there is no need to do both.
If the ``libcuda`` library is in a non-standard location, users can set
environment variable :envvar:`NUMBA_CUDA_DRIVER` to the file path (not the
directory path) of the shared library file.
