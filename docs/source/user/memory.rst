=================
Memory management
=================

.. _cuda-device-memory:

Data transfer
=============

Even though Numba can automatically transfer NumPy arrays to the device,
it can only do so conservatively by always transferring device memory back to
the host when a kernel finishes. To avoid the unnecessary transfer for
read-only arrays, you can use the following APIs to manually control the
transfer:

.. autofunction:: numba.cuda.device_array
   :noindex:
.. autofunction:: numba.cuda.device_array_like
   :noindex:
.. autofunction:: numba.cuda.to_device
   :noindex:

In addition to the device arrays, Numba can consume any object that implements
:ref:`cuda array interface <cuda-array-interface>`.  These objects also can be
manually converted into a Numba device array by creating a view of the GPU
buffer using the following APIs:

.. autofunction:: numba.cuda.as_cuda_array
  :noindex:
.. autofunction:: numba.cuda.is_cuda_array
  :noindex:


Device arrays
-------------

Device array references have the following methods.  These methods are to be
called in host code, not within CUDA-jitted functions.

.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :members: copy_to_host, is_c_contiguous, is_f_contiguous, ravel, reshape
    :noindex:


.. note:: DeviceNDArray defines the :ref:`cuda array interface <cuda-array-interface>`.


Pinned memory
=============

.. autofunction:: numba.cuda.pinned
   :noindex:
.. autofunction:: numba.cuda.pinned_array
   :noindex:
.. autofunction:: numba.cuda.pinned_array_like
   :noindex:


Mapped memory
=============

.. autofunction:: numba.cuda.mapped
   :noindex:
.. autofunction:: numba.cuda.mapped_array
   :noindex:
.. autofunction:: numba.cuda.mapped_array_like
   :noindex:


.. _cuda-managed-memory:

Managed memory
==============

.. autofunction:: numba.cuda.managed_array
   :noindex:


Streams
=======

Streams can be passed to functions that accept them (e.g. copies between the
host and device) and into kernel launch configurations so that the operations
are executed asynchronously.

.. autofunction:: numba.cuda.stream
   :noindex:

.. autofunction:: numba.cuda.default_stream
   :noindex:

.. autofunction:: numba.cuda.legacy_default_stream
   :noindex:

.. autofunction:: numba.cuda.per_thread_default_stream
   :noindex:

.. autofunction:: numba.cuda.external_stream
   :noindex:

CUDA streams have the following methods:

.. autoclass:: numba.cuda.cudadrv.driver.Stream
    :members: synchronize, auto_synchronize
    :noindex:

.. _cuda-shared-memory:

Shared memory and thread synchronization
========================================

A limited amount of shared memory can be allocated on the device to speed
up access to data, when necessary.  That memory will be shared (i.e. both
readable and writable) amongst all threads belonging to a given block
and has faster access times than regular device memory.  It also allows
threads to cooperate on a given solution.  You can think of it as a
manually-managed data cache.

The memory is allocated once for the duration of the kernel, unlike
traditional dynamic memory management.

.. function:: numba.cuda.shared.array(shape, type)
   :noindex:

   Allocate a shared array of the given *shape* and *type* on the device.
   This function must be called on the device (i.e. from a kernel or
   device function). *shape* is either an integer or a tuple of integers
   representing the array's dimensions and must be a simple constant
   expression. A "simple constant expression" includes, but is not limited to:

      #. A literal (e.g. ``10``)
      #. A local variable whose right-hand side is a literal or a simple constant
         expression (e.g. ``shape``, where ``shape`` is defined earlier in the function
         as ``shape = 10``)
      #. A global variable that is defined in the jitted function's globals by the time
         of compilation (e.g. ``shape``, where ``shape`` is defined using any expression
         at global scope).

   The definition must result in a Python ``int`` (i.e. not a NumPy scalar or other
   scalar / integer-like type). *type* is a :ref:`Numba type <numba-types>` of the
   elements needing to be stored in the array. The returned array-like object can be
   read and written to like any normal device array (e.g. through indexing).

   A common pattern is to have each thread populate one element in the
   shared array and then wait for all threads to finish using :func:`.syncthreads`.


.. function:: numba.cuda.syncthreads()
   :noindex:

   Synchronize all threads in the same thread block.  This function
   implements the same pattern as `barriers <http://en.wikipedia.org/wiki/Barrier_%28computer_science%29>`_
   in traditional multi-threaded programming: this function waits
   until all threads in the block call it, at which point it returns
   control to all its callers.

.. seealso::
   :ref:`Matrix multiplication example <cuda-matmul>`.

Dynamic Shared Memory
---------------------

In order to use dynamic shared memory in kernel code declare a shared array of
size 0:

.. code-block:: python

   @cuda.jit
   def kernel_func(x):
      dyn_arr = cuda.shared.array(0, dtype=np.float32)
      ...

and specify the size of dynamic shared memory in bytes during kernel invocation:

.. code-block:: python

   kernel_func[32, 32, 0, 128](x)

In the above code the kernel launch is configured with 4 parameters:

.. code-block:: python

   kernel_func[grid_dim, block_dim, stream, dyn_shared_mem_size]

**Note:** all dynamic shared memory arrays *alias*, so if you want to have
multiple dynamic shared arrays, you need to take *disjoint* views of the arrays.
For example, consider:

.. code-block:: python

   from numba import cuda
   import numpy as np

   @cuda.jit
   def f():
      f32_arr = cuda.shared.array(0, dtype=np.float32)
      i32_arr = cuda.shared.array(0, dtype=np.int32)
      f32_arr[0] = 3.14
      print(f32_arr[0])
      print(i32_arr[0])

   f[1, 1, 0, 4]()
   cuda.synchronize()

This allocates 4 bytes of shared memory (large enough for one ``int32`` or one
``float32``) and declares dynamic shared memory arrays of type ``int32`` and of
type ``float32``. When ``f32_arr[0]`` is set, this also sets the value of
``i32_arr[0]``, because they're pointing at the same memory. So we see as
output:

.. code-block:: pycon

   3.140000
   1078523331

because 1078523331 is the ``int32`` represented by the bits of the ``float32``
value 3.14.

If we take disjoint views of the dynamic shared memory:

.. code-block:: python

   from numba import cuda
   import numpy as np

   @cuda.jit
   def f_with_view():
      f32_arr = cuda.shared.array(0, dtype=np.float32)
      i32_arr = cuda.shared.array(0, dtype=np.int32)[1:] # 1 int32 = 4 bytes
      f32_arr[0] = 3.14
      i32_arr[0] = 1
      print(f32_arr[0])
      print(i32_arr[0])

   f_with_view[1, 1, 0, 8]()
   cuda.synchronize()

This time we declare 8 dynamic shared memory bytes, using the first 4 for a
``float32`` value and the next 4 for an ``int32`` value. Now we can set both the
``int32`` and ``float32`` value without them aliasing:

.. code-block:: pycon

   3.140000
   1


.. _cuda-local-memory:

Local memory
============

Local memory is an area of memory private to each thread.  Using local
memory helps allocate some scratchpad area when scalar local variables
are not enough.  The memory is allocated once for the duration of the kernel,
unlike traditional dynamic memory management.

.. function:: numba.cuda.local.array(shape, type)
   :noindex:

   Allocate a local array of the given *shape* and *type* on the device.
   *shape* is either an integer or a tuple of integers representing the array's
   dimensions and must be a simple constant expression. A "simple constant expression"
   includes, but is not limited to:

      #. A literal (e.g. ``10``)
      #. A local variable whose right-hand side is a literal or a simple constant
         expression (e.g. ``shape``, where ``shape`` is defined earlier in the function
         as ``shape = 10``)
      #. A global variable that is defined in the jitted function's globals by the time
         of compilation (e.g. ``shape``, where ``shape`` is defined using any expression
         at global scope).

   The definition must result in a Python ``int`` (i.e. not a NumPy scalar or other
   scalar / integer-like type). *type* is a :ref:`Numba type <numba-types>`
   of the elements needing to be stored in the array. The array is private to
   the current thread. An array-like object is returned which can be read and
   written to like any standard array (e.g. through indexing).

   .. seealso:: The Local Memory section of `Device Memory Accesses
      <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>`_
      in the CUDA programming guide.

Constant memory
===============

Constant memory is an area of memory that is read only, cached and off-chip, it
is accessible by all threads and is host allocated. A method of
creating an array in constant memory is through the use of:

.. function:: numba.cuda.const.array_like(arr)
   :noindex:

   Allocate and make accessible an array in constant memory based on array-like
   *arr*.


.. _deallocation-behavior:

Deallocation Behavior
=====================

This section describes the deallocation behaviour of Numba's internal memory
management. If an External Memory Management Plugin is in use (see
:ref:`cuda-emm-plugin`), then deallocation behaviour may differ; you may refer to the
documentation for the EMM Plugin to understand its deallocation behaviour.

Deallocation of all CUDA resources are tracked on a per-context basis.
When the last reference to a device memory is dropped, the underlying memory
is scheduled to be deallocated.  The deallocation does not occur immediately.
It is added to a queue of pending deallocations.  This design has two benefits:

1. Resource deallocation API may cause the device to synchronize; thus, breaking
   any asynchronous execution.  Deferring the deallocation could avoid latency
   in performance critical code section.
2. Some deallocation errors may cause all the remaining deallocations to fail.
   Continued deallocation errors can cause critical errors at the CUDA driver
   level.  In some cases, this could mean a segmentation fault in the CUDA
   driver. In the worst case, this could cause the system GUI to freeze and
   could only recover with a system reset.  When an error occurs during a
   deallocation, the remaining pending deallocations are cancelled.  Any
   deallocation error will be reported.  When the process is terminated, the
   CUDA driver is able to release all allocated resources by the terminated
   process.

The deallocation queue is flushed automatically as soon as the following events
occur:

- An allocation failed due to out-of-memory error.  Allocation is retried after
  flushing all deallocations.
- The deallocation queue has reached its maximum size, which is default to 10.
  User can override by setting the environment variable
  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT`.  For example,
  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT=20`, increases the limit to 20.
- The maximum accumulated byte size of resources that are pending deallocation
  is reached.  This is default to 20% of the device memory capacity.
  User can override by setting the environment variable
  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO`. For example,
  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO=0.5` sets the limit to 50% of the
  capacity.

Sometimes, it is desired to defer resource deallocation until a code section
ends.  Most often, users want to avoid any implicit synchronization due to
deallocation.  This can be done by using the following context manager:

.. autofunction:: numba.cuda.defer_cleanup
