..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


.. _cuda-call-conventions:

CUDA device call conventions
============================

Numba-CUDA supports two ABIs for device functions:

- The **Numba ABI**, used internally by Numba for most compiled device code.
- The **C ABI**, intended for interoperability with CUDA C/C++ style calls.

.. important::

   This is a major deviation from upstream Numba behavior: Numba-CUDA supports
   arbitrary nesting between these ABIs. A Numba-ABI function can call a
   C-ABI function, which can call a Numba-ABI function, and so on.


ABI overview
------------

Numba ABI
~~~~~~~~~

The Numba ABI is described in :ref:`device-function-abi` (without the
``extern "C"`` modifier):

- The function has a **status return code**.
- The Python return value is passed via a **pointer in the first argument**.
- Function names are mangled using Numba's mangling rules.
- Optional returns and exception status can be represented via the status
  channel.

C ABI
~~~~~

The C ABI behavior for compiled Python device functions is described in
:ref:`cuda-using-the-c-abi`:

- The function has a conventional C-style signature:
  ``<return_type>(<args...>)``.
- There is no separate status return code channel.
- Function names are predictable (by default the Python ``__name__``), and can
  be set explicitly with ``abi_info={"abi_name": ...}``.
- The C ABI is supported for device functions (not kernels).


Caller/callee matrix
--------------------

The table below summarizes what happens at each call edge:

.. list-table:: Caller and callee ABI combinations
   :header-rows: 1
   :widths: 22 39 39

   * - Caller / Callee
     - Numba ABI callee
     - C ABI callee
   * - Numba ABI caller
     - Numba-to-Numba call. Uses Numba ABI marshalling (status + return
       pointer), and propagates lower-frame error status.
     - Mixed call. Arguments / return are marshalled using the callee's C ABI
       signature. No callee status channel exists to propagate Python-exception
       status from the callee.
   * - C ABI caller
     - Mixed call. The call is marshalled using the callee's Numba ABI. The
       Numba callee can still produce status, but the C ABI caller has no
       outward status channel and does not propagate lower-frame status.
     - C-to-C call. Conventional C-style argument / return passing with no
       status channel.


What arbitrary nesting means
----------------------------

Each call site is lowered using the **callee's ABI**, not by forcing one ABI
for the whole call chain. This allows patterns like:

.. code:: text

   Numba ABI caller -> C ABI callee -> Numba ABI callee -> C ABI callee

to compile as expected.

In practice, this means mixed boundaries can appear at any depth in a call
graph, including calls to functions declared with
:func:`numba.cuda.declare_device` and calls to Numba-compiled device
subroutines.


Behavioral caveats
------------------

- The C ABI has no status channel for Python exception propagation.
- When a C ABI caller invokes a Numba ABI callee returning ``Optional[T]``,
  the optional is flattened to ``T`` at the C ABI boundary. A ``None`` result
  is represented as the default-initialized value of ``T``.
- Kernels must still use the Numba ABI entry model; compiling kernels with
  ``abi="c"`` is unsupported.
- For foreign CUDA C/C++ functions, use ``abi="c"`` with
  :func:`numba.cuda.declare_device` and follow pointer-signature guidance in
  :ref:`cuda_ffi`.
