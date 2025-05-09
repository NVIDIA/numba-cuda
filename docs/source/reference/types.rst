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

    Bfloat16 is a compute capability 8.0+ feature.

To determine whether Numba supports compiling code that uses the ``bfloat16``
type in the current configuration, use:

.. function:: numba.cuda.is_bfloat16_supported()

    Returns ``True`` if current device supports bfloat16. ``False`` otherwise.

Data Movement and Casts
***********************

Construction a single instance of bfloat16 object takes 1 API:

.. function:: numba.cuda.bf16.bfloat16(a)

    Constructs a bfloat16 data from existing device scalar. Supported scalar types:
    - float64
    - float32
    - int64
    - int32
    - uint64
    - uint32
    On cuda version 12.0+ machines, supports construction from ``float16``

Conversely, ``bfloat16`` data can be cast back to existing native data type via
``dtype(h)``, where ``dtype`` is one of the data types above (except float16),
and ``X`` is a bfloat16 object.

Arithmatics
************

A ``bfloat16`` data can be computed with another ``bfloat16`` data with many
supported aritheatic operators. The list of supported operations:

- Arithmatics (``+, -, *, /``)
- Arithmatic assignment oeprators (``+=, -=, *=, /=``)
- Logical operators (``==, !=, >, <, >=, <=``)
- Unary arithmatics (``+, -``)

Math Intrinsics
***************

A number of math intrinsics that utilizes the device native computing feature
on ``bfloat16`` are provided:

.. function:: htrunc(h)
    Round ``h`` to the nearest integer value that does not exceed ``h`` in magnitude.

.. function:: hceil(h)
    Compute the smallest integer value not less than ``h``.

.. function:: hfloor(h)
    Calculate the largest integer value which is less than or equal to ``h``.

.. function:: hrint(h)
    Round ``h`` to the nearest integer value in nv_bfloat16 floating-point format,
    with bfloat16way cases rounded to the nearest even integer value.

.. function:: hsqrt(a)
    Calculates bfloat16 square root of input ``a`` in round-to-nearest-even mode.

.. function:: hrsqrt(a)
    Calculates bfloat16 reciprocal square root of input ``a`` in round-to-nearest-even mode.

.. function:: hrcp(a)
   Calculates bfloat16 reciprocal of input a in round-to-nearest-even mode.

.. function:: hlog(a)
    Calculates bfloat16 natural logarithm of input ``a`` in round-to-nearest-even
    mode.

.. function:: hlog2(a)
    Calculates bfloat16 decimal logarithm of input ``a`` in round-to-nearest-even
    mode.

.. function:: hlog10(a)
    Calculates bfloat16 natural exponential function of input ``a`` in
    round-to-nearest-even mode.

.. function:: hcos(a)
    Calculates bfloat16 cosine of input ``a`` in round-to-nearest-even mode.

.. note::

    This function's implementation calls cosf(float) function and is exposed
    to compiler optimizations. Specifically, use_fast_math mode changes cosf(float)
    into an intrinsic __cosf(float), which has less accurate numeric behavior.

.. function:: hsin(a)
    Calculates bfloat16 sine of input ``a`` in round-to-nearest-even mode.

.. note::
    This function's implementation calls sinf(float) function and is exposed
    to compiler optimizations. Specifically, use_fast_math flag changes sinf(float)
    into an intrinsic __sinf(float), which has less accurate numeric behavior.

.. function:: hexp(a)
    Calculates bfloat16 natural exponential function of input ``a`` in
    round-to-nearest-even mode.

.. function:: hexp2(a)
    Calculates bfloat16 binary exponential function of input ``a`` in
    round-to-nearest-even mode.

.. function:: hexp10(h)
    Calculates bfloat16 decimal exponential function of input ``a`` in
    round-to-nearest-even mode.
