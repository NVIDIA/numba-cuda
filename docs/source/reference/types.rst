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
    - ``float16``

Conversely, ``bfloat16`` data can be cast back to existing native data type via
``dtype(b)``, where ``dtype`` is one of the data types above (except float16),
and ``b`` is a bfloat16 object.

Arithmetic
**********

Supported arithmetic operations on ``bfloat16`` operands are:

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
    Calculates bfloat16 binary logarithm (base-2) of input ``b`` in
    round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hlog10(b)
    Calculates bfloat16 common logarithm (base-10) of input ``b`` in
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


Arithmetic Intrinsics
*********************

The following low-level arithmetic intrinsics are available under
``numba.cuda.bf16`` and map to CUDA bfloat16 arithmetic functions. Unless
otherwise noted, operations are performed in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.habs(a)

    Calculates the absolute value of input ``a`` (bfloat16) and returns the result.

.. function:: numba.cuda.bf16.hneg(a)

    Negates input ``a`` (bfloat16) and returns the result.

.. function:: numba.cuda.bf16.hadd(a, b)

    Adds ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hadd_rn(a, b)

    Adds ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode. Prevents
    contraction of separate operations into a fused-multiply-add.

.. function:: numba.cuda.bf16.hadd_sat(a, b)

    Adds ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode, with
    saturation to the range ``[0.0, 1.0]``. NaN results are flushed to ``+0.0``.

.. function:: numba.cuda.bf16.hsub(a, b)

    Subtracts ``b`` from ``a`` (bfloat16) in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hsub_rn(a, b)

    Subtracts ``b`` from ``a`` (bfloat16) in round-to-nearest-even mode.
    Prevents contraction of separate operations into a fused-multiply-add.

.. function:: numba.cuda.bf16.hsub_sat(a, b)

    Subtracts ``b`` from ``a`` (bfloat16) in round-to-nearest-even mode, with
    saturation to the range ``[0.0, 1.0]``. NaN results are flushed to ``+0.0``.

.. function:: numba.cuda.bf16.hmul(a, b)

    Multiplies ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hmul_rn(a, b)

    Multiplies ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode.
    Prevents contraction of separate operations into a fused-multiply-add.

.. function:: numba.cuda.bf16.hmul_sat(a, b)

    Multiplies ``a`` and ``b`` (bfloat16) in round-to-nearest-even mode, with
    saturation to the range ``[0.0, 1.0]``. NaN results are flushed to ``+0.0``.

.. function:: numba.cuda.bf16.hdiv(a, b)

    Divides ``a`` by ``b`` (bfloat16) in round-to-nearest-even mode.

.. function:: numba.cuda.bf16.hfma(a, b, c)

    Computes a fused multiply-add of ``a`` and ``b`` plus ``c`` (bfloat16) in
    round-to-nearest-even mode; i.e. returns ``a * b + c``.

.. function:: numba.cuda.bf16.hfma_sat(a, b, c)

    Fused multiply-add in round-to-nearest-even mode with saturation to the
    range ``[0.0, 1.0]``. NaN results are flushed to ``+0.0``.

.. function:: numba.cuda.bf16.hfma_relu(a, b, c)

    Fused multiply-add in round-to-nearest-even mode with ReLU saturation;
    i.e. returns ``max(0, a * b + c)``.

Comparison Intrinsics
*********************

Device-level comparison intrinsics operating on ``bfloat16`` values are
available under ``numba.cuda.bf16``. Unless stated otherwise, the ordered
comparisons return ``False`` if either input is NaN, following IEEE semantics.

.. function:: numba.cuda.bf16.heq(a, b)

    Ordered equality. Returns ``True`` iff ``a == b``. NaN inputs yield ``False``.

.. function:: numba.cuda.bf16.hne(a, b)

    Ordered inequality. Returns ``True`` iff ``a != b`` and neither input is NaN.
    NaN inputs yield ``False``.

.. function:: numba.cuda.bf16.hge(a, b)

    Ordered greater-or-equal. NaN inputs yield ``False``.

.. function:: numba.cuda.bf16.hgt(a, b)

    Ordered greater-than. NaN inputs yield ``False``.

.. function:: numba.cuda.bf16.hle(a, b)

    Ordered less-or-equal. NaN inputs yield ``False``.

.. function:: numba.cuda.bf16.hlt(a, b)

    Ordered less-than. NaN inputs yield ``False``.

The unordered comparison variants return ``True`` when either input is NaN:

.. function:: numba.cuda.bf16.hequ(a, b)

    Unordered equality. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a == b``.

.. function:: numba.cuda.bf16.hneu(a, b)

    Unordered inequality. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a != b``.

.. function:: numba.cuda.bf16.hgeu(a, b)

    Unordered greater-or-equal. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a >= b``.

.. function:: numba.cuda.bf16.hgtu(a, b)

    Unordered greater-than. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a > b``.

.. function:: numba.cuda.bf16.hleu(a, b)

    Unordered less-or-equal. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a <= b``.

.. function:: numba.cuda.bf16.hltu(a, b)

    Unordered less-than. Returns ``True`` if ``a`` or ``b`` is NaN, or if ``a < b``.

Min/Max operations follow CUDA semantics for zeros and NaNs:

.. function:: numba.cuda.bf16.hmax(a, b)

    Returns ``max(a, b)`` with the following behavior:
    if either input is NaN, the other input is returned; if both are NaN,
    the canonical NaN is returned. If both inputs are zero, ``+0.0 > -0.0``.

.. function:: numba.cuda.bf16.hmin(a, b)

    Returns ``min(a, b)`` with the following behavior:
    if either input is NaN, the other input is returned; if both are NaN,
    the canonical NaN is returned. If both inputs are zero, ``+0.0 > -0.0``.

.. function:: numba.cuda.bf16.hmax_nan(a, b)

    Returns ``max(a, b)`` where NaNs pass through: if either input is NaN,
    the canonical NaN is returned.

.. function:: numba.cuda.bf16.hmin_nan(a, b)

    Returns ``min(a, b)`` where NaNs pass through: if either input is NaN,
    the canonical NaN is returned.

Special value predicates:

.. function:: numba.cuda.bf16.hisnan(a)

    Returns ``True`` if ``a`` is a NaN, ``False`` otherwise.

.. function:: numba.cuda.bf16.hisinf(a)

    Returns a nonzero integer if ``a`` is infinite, otherwise ``0``.

.. note::

    Python comparison operators on ``bfloat16`` values in device code map to
    the ordered comparisons above. For more details on the CUDA bfloat16
    comparison semantics, see `NVIDIA CUDA Math API: Bfloat16 Comparison Functions
    <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#group__cuda__math____bfloat16__comparison>`_.

Precision Conversion and Data Movement
*************************************

The following conversion intrinsics convert between ``bfloat16`` and other
scalar types. Rounding-mode suffixes:

- ``_rn``: round-to-nearest-even
- ``_rz``: round-towards-zero
- ``_rd``: round-down (towards −∞)
- ``_ru``: round-up (towards +∞)

Floating-point conversions
==========================

.. function:: numba.cuda.bf16.float32_to_bfloat16(x)

    Convert a ``float32`` to ``bfloat16`` (default rounding is round-to-nearest-even).

.. function:: numba.cuda.bf16.float64_to_bfloat16(x)

    Convert a ``float64`` to ``bfloat16`` (default rounding is round-to-nearest-even).

.. function:: numba.cuda.bf16.bfloat16_to_float32(x)

    Convert a ``bfloat16`` to ``float32``.

.. function:: numba.cuda.bf16.float32_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.float32_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.float32_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.float32_to_bfloat16_ru(x)

    Convert a ``float32`` to ``bfloat16`` using the specified rounding mode.

Integer conversions
===================

Representative APIs for each integer width are listed below. All have
rounding-mode variants ``_rn``, ``_rz``, ``_rd``, ``_ru``.

int16 (signed 16-bit)
---------------------

.. function:: numba.cuda.bf16.int16_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.int16_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.int16_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.int16_to_bfloat16_ru(x)

    Convert an ``int16`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_int16_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_int16_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_int16_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_int16_ru(x)

    Convert a ``bfloat16`` to ``int16`` with the selected rounding mode.

uint16 (unsigned 16-bit)
------------------------

.. function:: numba.cuda.bf16.uint16_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.uint16_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.uint16_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.uint16_to_bfloat16_ru(x)

    Convert a ``uint16`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_uint16_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint16_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint16_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint16_ru(x)

    Convert a ``bfloat16`` to ``uint16`` with the selected rounding mode.

int32 (signed 32-bit)
---------------------

.. function:: numba.cuda.bf16.int32_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.int32_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.int32_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.int32_to_bfloat16_ru(x)

    Convert an ``int32`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_int32_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_int32_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_int32_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_int32_ru(x)

    Convert a ``bfloat16`` to ``int32`` with the selected rounding mode.

uint32 (unsigned 32-bit)
------------------------

.. function:: numba.cuda.bf16.uint32_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.uint32_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.uint32_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.uint32_to_bfloat16_ru(x)

    Convert a ``uint32`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_uint32_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint32_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint32_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint32_ru(x)

    Convert a ``bfloat16`` to ``uint32`` with the selected rounding mode.

int64 (signed 64-bit)
---------------------

.. function:: numba.cuda.bf16.int64_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.int64_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.int64_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.int64_to_bfloat16_ru(x)

    Convert an ``int64`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_int64_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_int64_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_int64_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_int64_ru(x)

    Convert a ``bfloat16`` to ``int64`` with the selected rounding mode.

uint64 (unsigned 64-bit)
------------------------

.. function:: numba.cuda.bf16.uint64_to_bfloat16_rn(x)
.. function:: numba.cuda.bf16.uint64_to_bfloat16_rz(x)
.. function:: numba.cuda.bf16.uint64_to_bfloat16_rd(x)
.. function:: numba.cuda.bf16.uint64_to_bfloat16_ru(x)

    Convert a ``uint64`` to ``bfloat16`` with the selected rounding mode.

.. function:: numba.cuda.bf16.bfloat16_to_uint64_rn(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint64_rz(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint64_rd(x)
.. function:: numba.cuda.bf16.bfloat16_to_uint64_ru(x)

    Convert a ``bfloat16`` to ``uint64`` with the selected rounding mode.

8-bit conversions
=================

.. function:: numba.cuda.bf16.bfloat16_to_int8_rz(x)

    Convert a ``bfloat16`` to ``int8`` with round-towards-zero.

.. function:: numba.cuda.bf16.bfloat16_to_uint8_rz(x)

    Convert a ``bfloat16`` to ``uint8`` with round-towards-zero.

Bit Reinterpret Casts
*********************

These APIs reinterpret bits without numeric conversion:

.. function:: numba.cuda.bf16.bfloat16_as_int16(x)

    Reinterpret the bits of ``bfloat16`` as an ``int16``.

.. function:: numba.cuda.bf16.bfloat16_as_uint16(x)

    Reinterpret the bits of ``bfloat16`` as a ``uint16``.

.. function:: numba.cuda.bf16.int16_as_bfloat16(x)

    Reinterpret the bits of an ``int16`` as a ``bfloat16``.

.. function:: numba.cuda.bf16.uint16_as_bfloat16(x)

    Reinterpret the bits of a ``uint16`` as a ``bfloat16``.
