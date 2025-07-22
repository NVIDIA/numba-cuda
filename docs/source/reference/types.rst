CUDA-Specific Types
====================

.. note::

    This page is about types specific to CUDA targets. Many other types are also
    available in the CUDA target - see :ref:`cuda-built-in-types`.

Vector Types
~~~~~~~~~~~~

`CUDA Vector Types <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types>`_
are usable in kernels. There are two important distinctions from vector types in CUDA C/C++:

First, the recommended names for vector types in Numba CUDA is formatted as ``<base_type>x<N>``,
where ``base_type`` is the base type of the vector, and ``N`` is the number of elements in the vector.
Examples include ``int64x3``, ``uint16x4``, ``float32x4``, etc. For new Numba CUDA kernels,
this is the recommended way to instantiate vector types.

For convenience, users adapting existing kernels from CUDA C/C++ to Python may use
aliases consistent with the C/C++ namings. For example, ``float3`` aliases ``float32x3``,
``long3`` aliases ``int32x3`` or ``int64x3`` (depending on the platform), etc.

Second, unlike CUDA C/C++ where factory functions are used, vector types are constructed directly
with their constructor. For example, to construct a ``float32x3``:

.. code-block:: python3

    from numba.cuda import float32x3

    # In kernel
    f3 = float32x3(0.0, -1.0, 1.0)

Additionally, vector types can be constructed from a combination of vector and
primitive types, as long as the total number of components matches the result
vector type. For example, all of the following constructions are valid:

.. code-block:: python3

    zero = uint32(0)
    u2 = uint32x2(1, 2)
    # Construct a 3-component vector with primitive type and a 2-component vector
    u3 = uint32x3(zero, u2)
    # Construct a 4-component vector with 2 2-component vectors
    u4 = uint32x4(u2, u2)

The 1st, 2nd, 3rd and 4th component of the vector type can be accessed through fields
``x``, ``y``, ``z``, and ``w`` respectively. The components are immutable after
construction in the present version of Numba; it is expected that support for
mutating vector components will be added in a future release.

.. code-block:: python3

    v1 = float32x2(1.0, 1.0)
    v2 = float32x2(1.0, -1.0)
    dotprod = v1.x * v2.x + v1.y * v2.y


Narrow Data Types
~~~~~~~~~~~~~~~~~

Bfloat16
--------

.. note::

    Bfloat16 is only supported with CUDA version 12.0+, and only supported on
    devices with compute capability 8.0 or above.

To determine whether ``bfloat16`` is supported in the current configuration,
use:

.. function:: numba.cuda.is_bfloat16_supported()

    Returns ``True`` if the current device and toolkit support bfloat16.
    ``False`` otherwise.

Data Movement and Casts
***********************

Construction of a single instance of a ``bfloat16`` object:

.. function:: numba.cuda.bf16.bfloat16(b)

    Constructs a ``bfloat16`` from existing device `scalar`. Supported scalar
    types:

    - ``float64``
    - ``float32``
    - ``float16``
    - ``int64``
    - ``int32``
    - ``uint64``
    - ``uint32``

Conversely, ``bfloat16`` data can be cast back to existing native data type via
``dtype(b)``, where ``dtype`` is one of the data types above (except float16),
and ``b`` is a bfloat16 object.

Arithmetic
**********

Supported arithmetic operations on ``bfloat`16`` operands are:

- Arithmetic (``+``, ``-``, ``*``, ``/``)
- Arithmetic assignment operators (``+=``, ``-=``, ``*=``, ``/=``)
- Logical operators (``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``)
- Unary arithmetic (``+``, ``-``)

Math Intrinsics
***************

A number of math intrinsics that utilizes the device native computing feature
on ``bfloat16`` are provided:

.. function:: numba.cuda.bf16.htrunc(b)
    Round ``b`` to the nearest integer value that does not exceed ``b`` in magnitude.

.. function:: numba.cuda.bf16.hceil(b)
    Compute the smallest integer value not less than ``b``.

.. function:: numba.cuda.bf16.hfloor(b)
    Calculate the largest integer value which is less than or equal to ``b``.

.. function:: numba.cuda.bf16.hrint(b)
    Round ``b`` to the nearest integer value in nv_bfloat16 floating-point format,
    with halfway cases rounded to the nearest even integer value.

.. function:: numba.cuda.bf16.hsqrt(b)
    Calculates bfloat16 square root of input ``b`` in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hrsqrt(b)
    Calculates bfloat16 reciprocal square root of input ``b`` in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hrcp(b)
   Calculates bfloat16 reciprocal of input a in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hlog(b)
    Calculates bfloat16 natural logarithm of input ``b`` in round-to-nearest-even
    mode.

.. function:: numba.cuda.bf16.hlog2(b)
    Calculates bfloat16 decimal logarithm of input ``b`` in round-to-nearest-even
    mode.

.. function:: numba.cuda.bf16.hlog10(b)
    Calculates bfloat16 natural exponential function of input ``b`` in
    round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hcos(b)
    Calculates bfloat16 cosine of input ``b`` in round-to-nearest-even mode.

.. note::

    This function's implementation calls cosf(float) function and is exposed
    to compiler optimizations. Specifically, use_fast_math mode changes cosf(float)
    into an intrinsic __cosf(float), which has less accurate numeric behavior.

.. function:: numba.cuda.bf16.hsin(b)
    Calculates bfloat16 sine of input ``b`` in round-to-nearest-even mode.

.. note::
    This function's implementation calls sinf(float) function and is exposed
    to compiler optimizations. Specifically, use_fast_math flag changes sinf(float)
    into an intrinsic __sinf(float), which has less accurate numeric behavior.

.. function:: numba.cuda.bf16.htanh(b)
    Calculates bfloat16 hyperbolic tangent function: ``tanh(b)`` in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.htanh_approx(b)
    Calculates approximate bfloat16 hyperbolic tangent function: ``tanh(b)``.
    This operation uses HW acceleration on devices of compute capability 9.x and higher.

.. note::
    tanh_approx(0)      returns 0
    tanh_approx(inf)    returns 1
    tanh_approx(nan)    returns nan

.. function:: numba.cuda.bf16.hexp(b)
    Calculates bfloat16 natural exponential function of input ``b`` in
    round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hexp2(b)
    Calculates bfloat16 binary exponential function of input ``b`` in
    round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hexp10(b)
    Calculates bfloat16 decimal exponential function of input ``b`` in
    round-to-nearest-even mode.
