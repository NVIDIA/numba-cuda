..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


.. _cuda_ffi:

Calling foreign functions from Python kernels
=============================================

Python kernels can call device functions written in other languages. CUDA C/C++,
PTX, and binary objects (cubins, fat binaries, etc.) are directly supported;
sources in other languages must be compiled to PTX first. The constituent parts
of a Python kernel call to a foreign device function are:

- The device function implementation in a foreign language (e.g. CUDA C).
- A declaration of the device function in Python.
- A kernel that calls the foreign function.

.. _device-function-abi:

Device function ABI
-------------------

Numba's ABI for calling device functions defines the following prototype in
C/C++:

.. code:: C

   extern "C"
   __device__ int
   function(
     T* return_value,
     ...
   );


Components of the prototype are as follows:

- ``extern "C"`` is used to prevent name-mangling so that it is easy to declare
  the function in Python. It can be removed, but then the mangled name must be
  used in the declaration of the function in Python.
- ``__device__`` is required to define the function as a device function.
- The return value is always of type ``int``, and is used to signal whether a
  Python exception occurred. Since Python exceptions don't occur in foreign
  functions, this should always be set to 0 by the callee.
- The first argument is a pointer to the return value of type ``T``, which is
  allocated in the local address space [#f1]_ and passed in by the caller. If
  the function returns a value, the pointee should be set by the callee to
  store the return value.
- Subsequent arguments should match the types and order of arguments passed to
  the function from the Python kernel.

Functions written in other languages must compile to PTX that conforms to this
prototype specification.

A function that accepts two floats and returns a float would have the following
prototype:

.. code:: C

   extern "C"
   __device__ int
   mul_f32_f32(
     float* return_value,
     float x,
     float y
   );

.. rubric:: Notes

.. [#f1] Care must be taken to ensure that any operations on the return value
         are applicable to data in the local address space.  Some operations,
         such as atomics, cannot be performed on data in the local address
         space.

Declaration in Python
---------------------

To declare a foreign device function in Python, use :func:`declare_device()
<numba.cuda.declare_device>`:

.. autofunction:: numba.cuda.declare_device

The returned descriptor name need not match the name of the foreign function.
For example, when:

.. code::

   mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)' , link="functions.cu")

is declared, calling ``mul(a, b)`` inside a kernel will translate into a call to
``mul_f32_f32(a, b)`` in the compiled code.

Passing pointers
----------------

Numba's calling convention requires multiple values to be passed for array
arguments. These include the data pointer along with shape, stride, and other
information. This is incompatible with the expectations of most C/C++ functions,
which generally only expect a pointer to the data. To align the calling
conventions between C device code and Python kernels it is necessary to declare
array arguments using C pointer types.

For example, a function with the following prototype:

.. literalinclude:: ../../../tests/doc_examples/ffi/functions.cu
   :language: C
   :caption: ``tests/doc_examples/ffi/functions.cu``
   :start-after: magictoken.ex_sum_reduce_proto.begin
   :end-before: magictoken.ex_sum_reduce_proto.end
   :linenos:

would be declared as follows:

.. literalinclude:: ../../../tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_from_buffer`` in ``tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_from_buffer_decl.begin
   :end-before: magictoken.ex_from_buffer_decl.end
   :dedent: 8
   :linenos:

To obtain a pointer to array data for passing to foreign functions, use the
``from_buffer()`` method of a ``cffi.FFI`` instance. For example, a kernel using
the ``sum_reduce`` function could be defined as:

.. literalinclude:: ../../../tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_from_buffer`` in ``tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_from_buffer_kernel.begin
   :end-before: magictoken.ex_from_buffer_kernel.end
   :dedent: 8
   :linenos:

where ``result`` and ``array`` are both arrays of ``float32`` data.

Linking and Calling functions
-----------------------------

The ``link`` keyword argument to the :func:`declare_device
<numba.cuda.declare_device>` function accepts *Linkable Code* items. Either a
single Linkable Code item can be passed, or multiple items in a list, tuple, or
set.

A Linkable Code item is either:

* A string indicating the location of a file in the filesystem, or
* A :class:`LinkableCode <numba.cuda.LinkableCode>` object, for linking code
  that exists in memory.

Suported code formats that can be linked are:

* PTX source code (``*.ptx``)
* CUDA C/C++ source code (``*.cu``)
* CUDA ELF Fat Binaries (``*.fatbin``)
* CUDA ELF Cubins (``*.cubin``)
* CUDA ELF archives (``*.a``)
* CUDA Object files (``*.o``)
* CUDA LTOIR files (``*.ltoir``)

CUDA C/C++ source code will be compiled with the `NVIDIA Runtime Compiler
(NVRTC) <https://docs.nvidia.com/cuda/nvrtc/index.html>`_ and linked into the
kernel as either PTX or LTOIR, depending on whether LTO is enabled. Other files
will be passed directly to the CUDA Linker.

A ``LinkableCode`` object may have setup and teardown callback functions that
perform module-specific initialization and cleanup tasks.

* Setup functions are invoked once for every new module loaded.
* Teardown functions are invoked just prior to module unloading.

Both setup and teardown callbacks are called with a handle to the relevant
module. In practice, Numba creates a new module each time a kernel is compiled
for a specific set of argument types.

For each module, the setup callback is invoked once only. When a module is
executed by multiple threads, only one thread will execute the setup
callback.

The callbacks are defined as follows:

.. code::

  def setup_callback(mod: cuda.cudadrv.drvapi.cu_module):...
  def teardown_callback(mod: cuda.cudadrv.drvapi.cu_module):...

:class:`LinkableCode <numba.cuda.LinkableCode>` objects are initialized using
the parameters of their base class:

.. autoclass:: numba.cuda.LinkableCode

However, one should instantiate an instance of the class that represents the
type of item being linked:

.. autoclass:: numba.cuda.PTXSource
.. autoclass:: numba.cuda.CUSource
.. autoclass:: numba.cuda.Fatbin
.. autoclass:: numba.cuda.Cubin
.. autoclass:: numba.cuda.Archive
.. autoclass:: numba.cuda.Object
.. autoclass:: numba.cuda.LTOIR

Legacy ``@cuda.jit`` decorator ``link`` support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``link`` keyword argument of the :func:`@cuda.jit <numba.cuda.jit>`
decorator also accepts a list of Linkable Code items, which will then be linked
into the kernel. This facility is provided for backwards compatibility; it is
recommended that Linkable Code items are always specified in the
:func:`declare_device <numba.cuda.declare_device>` call, so that the user of the
declared API is not burdened with specifying the items to link themselves when
writing a kernel.

As an example of how this legacy mechanism looked at the point of use: the
following kernel calls the ``mul()`` function declared above with the
implementation ``mul_f32_f32()`` as if it were in a file called ``functions.cu``
that had not been declared as part of the ``link`` argument in the declaration:

.. code::

   @cuda.jit(link=['functions.cu'])
   def multiply_vectors(r, x, y):
       i = cuda.grid(1)

       if i < len(r):
           r[i] = mul(x[i], y[i])

C/C++ Support
-------------

Support for compiling and linking of CUDA C/C++ code is provided through the use
of NVRTC subject to the following considerations:

- A suitable version of the NVRTC library must be available.
- The CUDA include path is assumed by default to be ``/usr/local/cuda/include``
  on Linux and ``$env:CUDA_PATH\include`` on Windows. It can be modified using
  the environment variable :envvar:`NUMBA_CUDA_INCLUDE_PATH`.
- The CUDA include directory will be made available to NVRTC on the include
  path.
- Additional search paths can be set to the environment variable
  :envvar:`NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS`. Multiple paths should be colon
  separated.

Extra Search Paths Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates calling a foreign function that includes additional
headers not in the default Numba-CUDA search paths.

The definitions of the C++ template APIs are in two different locations:

.. literalinclude:: ../../../tests/doc_examples/ffi/include/mul.cuh
   :language: C
   :caption: ``tests/doc_examples/ffi/include/mul.cuh``
   :linenos:

.. literalinclude:: ../../../tests/data/include/add.cuh
   :language: C
   :caption: ``tests/data/include/add.cuh``
   :linenos:

Neither of the headers are in the default search paths of Numba-CUDA, but the
foreign device function ``saxpy`` depends on them:

.. literalinclude:: ../../../tests/doc_examples/ffi/saxpy.cu
   :language: C
   :caption: ``tests/data/doc_examples/ffi/saxpy.cu``
   :linenos:

In the Python code, assume that ``mul_dir`` and ``add_dir`` are set to the
paths that contain ``mul.cuh`` and ``add.cuh`` respectively. The paths are
joined with ``:`` before setting ``config.CUDA_NVRTC_EXTRA_SEARCH_PATHS``:

.. literalinclude:: ../../../tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_extra_includes`` in ``tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_extra_search_paths.begin
   :end-before: magictoken.ex_extra_search_paths.end
   :dedent: 12
   :linenos:

Next, use ``saxpy`` as intended:

.. literalinclude:: ../../../tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_extra_includes`` in ``tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_extra_search_paths_kernel.begin
   :end-before: magictoken.ex_extra_search_paths_kernel.end
   :dedent: 12
   :linenos:


Complete Example
----------------

This example demonstrates calling a foreign function written in CUDA C to
multiply pairs of numbers from two arrays.

The foreign function is written as follows:

.. literalinclude:: ../../../tests/doc_examples/ffi/functions.cu
   :language: C
   :caption: ``tests/doc_examples/ffi/functions.cu``
   :start-after: magictoken.ex_mul_f32_f32.begin
   :end-before: magictoken.ex_mul_f32_f32.end
   :linenos:

The Python code and kernel are:

.. literalinclude:: ../../../tests/doc_examples/test_ffi.py
   :language: python
   :caption: from ``test_ex_linking_cu`` in ``tests/doc_examples/test_ffi.py``
   :start-after: magictoken.ex_linking_cu.begin
   :end-before: magictoken.ex_linking_cu.end
   :dedent: 8
   :linenos:

.. note::

  The example above is minimal in order to illustrate a foreign function call -
  it would not be expected to be particularly performant due to the small grid
  and light workload of the foreign function.
