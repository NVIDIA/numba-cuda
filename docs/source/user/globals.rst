..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


.. _cuda-globals:

=====================================
Global Variables and Captured Values
=====================================

Numba CUDA kernels and device functions can reference global variables defined
at module scope. This section describes how these values are captured and the
implications for your code.


Capture as constants
====================

By default, global variables referenced in kernels are captured as constants at
compilation time. This applies to scalars and host arrays (e.g. NumPy arrays).

The following example demonstrates this behavior. Both ``TAX_RATE`` and
``PRICES`` are captured when the kernel is first compiled. Because they are
embedded as constants, **modifications to these variables after compilation
have no effect**—the second kernel call still uses the original values:

.. literalinclude:: ../../../numba_cuda/numba/cuda/tests/doc_examples/test_globals.py
   :language: python
   :caption: Demonstrating constant capture of global variables
   :start-after: magictoken.ex_globals_constant_capture.begin
   :end-before: magictoken.ex_globals_constant_capture.end
   :dedent: 8
   :linenos:

Running the above code prints:

.. code-block:: text

   Value of d_totals: [ 10.8  54.   16.2  64.8 162. ]
   Value of d_totals: [ 10.8  54.   16.2  64.8 162. ]

Note that both outputs are identical—the modifications to ``TAX_RATE`` and
``PRICES`` after the first kernel call have no effect.

This behaviour is useful for small amounts of truly constant data like
configuration values, lookup tables, or mathematical constants. For larger
arrays, consider using device arrays instead.


Device array capture
====================

Device arrays are an exception to the constant capture rule. When a kernel
references a global device array—any object implementing
``__cuda_array_interface__``, such as CuPy arrays or Numba device arrays—the
device pointer is captured rather than the data. No copy occurs, and
modifications to the array **are** visible to subsequent kernel calls.

The following example demonstrates this behavior. The global ``PRICES`` device
array is mutated after the first kernel call, and the second kernel call sees
the updated values:

.. literalinclude:: ../../../numba_cuda/numba/cuda/tests/doc_examples/test_globals.py
   :language: python
   :caption: Demonstrating device array capture by pointer
   :start-after: magictoken.ex_globals_device_array_capture.begin
   :end-before: magictoken.ex_globals_device_array_capture.end
   :dedent: 8
   :linenos:

Running the above code prints:

.. code-block:: text

   [10. 25.  5. 15. 30.]
   [20. 50. 10. 30. 60.]

Note that the outputs are different—the mutation to ``PRICES`` after the first
kernel call *is* visible to the second call, unlike with host arrays.

This makes device arrays suitable for global state that needs to be updated
between kernel calls without recompilation.

.. note::

   Kernels and device functions that capture global device arrays cannot use
   ``cache=True``. Because the device pointer is embedded in the compiled code,
   caching would serialize an invalid pointer. Attempting to cache such a kernel
   will raise a ``PicklingError``. See :doc:`caching` for more information on
   kernel caching.
