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

.. code-block:: python

   import numpy as np
   from numba import cuda

   TAX_RATE = 0.08
   PRICES = np.array([10.0, 25.0, 5.0, 15.0, 30.0], dtype=np.float64)

   @cuda.jit
   def compute_totals(quantities, totals):
       i = cuda.grid(1)
       if i < totals.size:
           totals[i] = quantities[i] * PRICES[i] * (1 + TAX_RATE)

Both ``TAX_RATE`` and ``PRICES`` are captured when the kernel is compiled.
Because they are embedded as constants, **modifications to these variables
after compilation have no effect**:

.. code-block:: python

   # First kernel call - compiles and captures values
   compute_totals[1, 32](d_quantities, d_totals)

   # These modifications have no effect on subsequent kernel calls
   TAX_RATE = 0.10
   PRICES[:] = [20.0, 50.0, 10.0, 30.0, 60.0]

   # Second kernel call still uses the original values
   compute_totals[1, 32](d_quantities, d_totals)

This behaviour is useful for small amounts of truly constant data like
configuration values, lookup tables, or mathematical constants. For larger
arrays, consider using device arrays instead.


Device array capture
====================

Device arrays are an exception to the constant capture rule. When a kernel
references a global device array—any object implementing
``__cuda_array_interface__``, such as CuPy arrays or Numba device arrays—the
device pointer is captured rather than the data. No copy occurs, and
modifications to the array **are** visible to subsequent kernel calls:

.. code-block:: python

   import numpy as np
   import cupy as cp
   from numba import cuda

   # Global device array - pointer is captured, not data
   PRICES = cp.array([10.0, 25.0, 5.0, 15.0, 30.0], dtype=np.float32)

   @cuda.jit
   def compute_totals(quantities, totals):
       i = cuda.grid(1)
       if i < totals.size:
           totals[i] = quantities[i] * PRICES[i]

   # First kernel call
   compute_totals[1, 32](d_quantities, d_totals)
   print(d_totals.copy_to_host())  # [10. 25.  5. 15. 30.]

   # Mutate the device array
   PRICES[:] = cp.array([20.0, 50.0, 10.0, 30.0, 60.0], dtype=np.float32)

   # Second kernel call sees the updated values
   compute_totals[1, 32](d_quantities, d_totals)
   print(d_totals.copy_to_host())  # [20. 50. 10. 30. 60.]

This makes device arrays suitable for global state that needs to be updated
between kernel calls without recompilation.

.. note::

   Kernels and device functions that capture global device arrays cannot use
   ``cache=True``. Because the device pointer is embedded in the compiled code,
   caching would serialize an invalid pointer. Attempting to cache such a kernel
   will raise a ``PicklingError``. See :doc:`caching` for more information on
   kernel caching.
