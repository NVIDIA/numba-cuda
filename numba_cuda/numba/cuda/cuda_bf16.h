/*
* Copyright 1993-2024 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

/**
* \defgroup CUDA_MATH_INTRINSIC_BFLOAT16 Bfloat16 Precision Intrinsics
* This section describes nv_bfloat16 precision intrinsic functions.
* To use these functions, include the header file \p cuda_bf16.h in your program.
* All of the functions defined here are available in device code.
* Some of the functions are also available to host compilers, please
* refer to respective functions' documentation for details.
*
* NOTE: Aggressive floating-point optimizations performed by host or device
* compilers may affect numeric behavior of the functions implemented in this
* header. Specific examples are:
* - hsin(__nv_bfloat16);
* - hcos(__nv_bfloat16);
* - h2sin(__nv_bfloat162);
* - h2cos(__nv_bfloat162);
*
* The following macros are available to help users selectively enable/disable
* various definitions present in the header file:
* - \p CUDA_NO_BFLOAT16 - If defined, this macro will prevent the definition of
* additional type aliases in the global namespace, helping to avoid potential
* conflicts with symbols defined in the user program.
* - \p __CUDA_NO_BFLOAT16_CONVERSIONS__ - If defined, this macro will prevent
* the use of the C++ type conversions (converting constructors and conversion
* operators) that are common for built-in floating-point types, but may be
* undesirable for \p __nv_bfloat16 which is essentially a user-defined type.
* - \p __CUDA_NO_BFLOAT16_OPERATORS__ and \p __CUDA_NO_BFLOAT162_OPERATORS__ -
* If defined, these macros will prevent the inadvertent use of usual arithmetic
* and comparison operators. This enforces the storage-only type semantics and
* prevents C++ style computations on \p __nv_bfloat16 and \p __nv_bfloat162 types.
*/

/**
* \defgroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS Bfloat16 Arithmetic Constants
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these constants, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT16_ARITHMETIC Bfloat16 Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT162_ARITHMETIC Bfloat162 Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT16_COMPARISON Bfloat16 Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT162_COMPARISON Bfloat162 Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT16_MISC Bfloat16 Precision Conversion and Data Movement
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT16_FUNCTIONS Bfloat16 Math Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

/**
* \defgroup CUDA_MATH__BFLOAT162_FUNCTIONS Bfloat162 Math Functions
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
* To use these functions, include the header file \p cuda_bf16.h in your program.
*/

#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__

/* bring in __half data type and operations, for use in converting constructors */
#include "cuda_fp16.h"

// implicitly provided by NVRTC
#if !defined(__CUDACC_RTC__)
/* bring in float2, double4, etc vector types */
#include "vector_types.h"
/* bring in operations on vector types like: make_float2 */
#include "vector_functions.h"
#endif  /* !defined(__CUDACC_RTC__) */

#define ___CUDA_BF16_STRINGIFY_INNERMOST(x) #x
#define __CUDA_BF16_STRINGIFY(x) ___CUDA_BF16_STRINGIFY_INNERMOST(x)

#if defined(__cplusplus)

/* Set up function decorations */
#if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))
#define __CUDA_BF16_DECL__ __device__
#define __CUDA_HOSTDEVICE_BF16_DECL__ __device__
#define __CUDA_HOSTDEVICE__ __device__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define __CUDA_BF16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_BF16_DECL__ static __host__ __device__ __inline__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
#if defined(__GNUC__)
#define __CUDA_HOSTDEVICE_BF16_DECL__ static __attribute__ ((unused))
#else
#define __CUDA_HOSTDEVICE_BF16_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE__
#endif /* (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))) */

#define __CUDA_BF16_TYPES_EXIST__

/* Macros to allow nv_bfloat16 & nv_bfloat162 to be used by inline assembly */
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __BFLOAT16_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __BFLOAT162_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __BFLOAT162_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

/* Forward-declaration of structures defined in "cuda_bf16.hpp" */
struct __nv_bfloat16;
struct __nv_bfloat162;

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts double number to nv_bfloat16 precision in round-to-nearest-even mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts double number \p a to nv_bfloat16 precision in round-to-nearest-even mode.
* \param[in] a - double. Is only being read.
* \returns nv_bfloat16
* - \p a converted to \p nv_bfloat16 using round-to-nearest-even mode.
* - __double2bfloat16 \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __double2bfloat16 \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __double2bfloat16(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __double2bfloat16(const double a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts float number to nv_bfloat16 precision in round-to-nearest-even mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts float number \p a to nv_bfloat16 precision in round-to-nearest-even mode.
* \param[in] a - float. Is only being read.
* \returns nv_bfloat16
* - \p a converted to nv_bfloat16 using round-to-nearest-even mode.
*
* \see __float2bfloat16_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts float number to nv_bfloat16 precision in round-to-nearest-even mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts float number \p a to nv_bfloat16 precision in round-to-nearest-even mode.
* \param[in] a - float. Is only being read.
* \returns nv_bfloat16
* - \p a converted to nv_bfloat16 using round-to-nearest-even mode.
* - __float2bfloat16_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2bfloat16_rn \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2bfloat16_rn(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rn(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts float number to nv_bfloat16 precision in round-towards-zero mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts float number \p a to nv_bfloat16 precision in round-towards-zero mode.
* \param[in] a - float. Is only being read.
* \returns nv_bfloat16
* - \p a converted to nv_bfloat16 using round-towards-zero mode.
* - __float2bfloat16_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2bfloat16_rz \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2bfloat16_rz(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rz(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts float number to nv_bfloat16 precision in round-down mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts float number \p a to nv_bfloat16 precision in round-down mode.
* \param[in] a - float. Is only being read.
*
* \returns nv_bfloat16
* - \p a converted to nv_bfloat16 using round-down mode.
* - __float2bfloat16_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2bfloat16_rd \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2bfloat16_rd(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rd(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts float number to nv_bfloat16 precision in round-up mode
* and returns \p nv_bfloat16 with converted value.
*
* \details Converts float number \p a to nv_bfloat16 precision in round-up mode.
* \param[in] a - float. Is only being read.
*
* \returns nv_bfloat16
* - \p a converted to nv_bfloat16 using round-up mode.
* - __float2bfloat16_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __float2bfloat16_ru \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __float2bfloat16_ru(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_ru(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts \p nv_bfloat16 number to float.
*
* \details Converts nv_bfloat16 number \p a to float.
* \param[in] a - float. Is only being read.
*
* \returns float
* - \p a converted to float.
* - __bfloat162float \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
* - __bfloat162float \cuda_math_formula (\pm \infty)\end_cuda_math_formula returns \cuda_math_formula \pm \infty \end_cuda_math_formula.
* - __bfloat162float(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ float __bfloat162float(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts input to nv_bfloat16 precision in round-to-nearest-even mode and
* populates both halves of \p nv_bfloat162 with converted value.
*
* \details Converts input \p a to nv_bfloat16 precision in round-to-nearest-even mode and
* populates both halves of \p nv_bfloat162 with converted value.
* \param[in] a - float. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 value with both halves equal to the converted nv_bfloat16
* precision number.
*
* \see __float2bfloat16_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float2bfloat162_rn(const float a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts both input floats to nv_bfloat16 precision in round-to-nearest-even
* mode and returns \p nv_bfloat162 with converted values.
*
* \details Converts both input floats to nv_bfloat16 precision in round-to-nearest-even mode
* and combines the results into one \p nv_bfloat162 number. Low 16 bits of the return
* value correspond to the input \p a, high 16 bits correspond to the input \p
* b.
* \param[in] a - float. Is only being read.
* \param[in] b - float. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 value with corresponding halves equal to the
* converted input floats.
*
* \see __float2bfloat16_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts low 16 bits of \p nv_bfloat162 to float and returns the result
*
* \details Converts low 16 bits of \p nv_bfloat162 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns float
* - The low 16 bits of \p a converted to float.
*
* \see __bfloat162float(__nv_bfloat16) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ float __low2float(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts high 16 bits of \p nv_bfloat162 to float and returns the result
*
* \details Converts high 16 bits of \p nv_bfloat162 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns float
* - The high 16 bits of \p a converted to float.
*
* \see __bfloat162float(__nv_bfloat16) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ float __high2float(const __nv_bfloat162 a);

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts both components of float2 number to nv_bfloat16 precision in
* round-to-nearest-even mode and returns \p nv_bfloat162 with converted values.
*
* \details Converts both components of float2 to nv_bfloat16 precision in round-to-nearest-even
* mode and combines the results into one \p nv_bfloat162 number. Low 16 bits of the
* return value correspond to \p a.x and high 16 bits of the return value
* correspond to \p a.y.
* \param[in] a - float2. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 which has corresponding halves equal to the
* converted float2 components.
*
* \see __float2bfloat16_rn(float) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float22bfloat162_rn(const float2 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Converts both halves of \p nv_bfloat162 to float2 and returns the result.
*
* \details Converts both halves of \p nv_bfloat162 input \p a to float and returns the
* result as a \p float2 packed value.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns float2
* - \p a converted to float2.
*
* \see __bfloat162float(__nv_bfloat16) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ float2 __bfloat1622float2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed char in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed
* char in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns signed char
* - \p h converted to a signed char using round-towards-zero mode.
* - __bfloat162char_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162char_rz \cuda_math_formula (x), x > 127\end_cuda_math_formula returns SCHAR_MAX = \p 0x7F.
* - __bfloat162char_rz \cuda_math_formula (x), x < -128\end_cuda_math_formula returns SCHAR_MIN = \p 0x80.
* - __bfloat162char_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ signed char __bfloat162char_rz(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned char in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned
* char in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned char
* - \p h converted to an unsigned char using round-towards-zero mode.
* - __bfloat162uchar_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162uchar_rz \cuda_math_formula (x), x > 255\end_cuda_math_formula returns UCHAR_MAX = \p 0xFF.
* - __bfloat162uchar_rz \cuda_math_formula (x), x < 0.0\end_cuda_math_formula returns 0.
* - __bfloat162uchar_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned char __bfloat162uchar_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed integer in round-to-nearest-even mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed integer in
* round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns int
* - \p h converted to a signed integer using round-to-nearest-even mode.
* - __bfloat162int_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162int_rn \cuda_math_formula (x), x > INT_MAX\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __bfloat162int_rn \cuda_math_formula (x), x < INT_MIN\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __bfloat162int_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ int __bfloat162int_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed integer in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed integer in
* round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns int
* - \p h converted to a signed integer using round-towards-zero mode.
* - __bfloat162int_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162int_rz \cuda_math_formula (x), x > INT_MAX\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __bfloat162int_rz \cuda_math_formula (x), x < INT_MIN\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __bfloat162int_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ int __bfloat162int_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed integer in
* round-down mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns int
* - \p h converted to a signed integer using round-down mode.
* - __bfloat162int_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162int_rd \cuda_math_formula (x), x > INT_MAX\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __bfloat162int_rd \cuda_math_formula (x), x < INT_MIN\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __bfloat162int_rd(NaN) returns 0.* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ int __bfloat162int_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed integer in
* round-up mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns int
* - \p h converted to a signed integer using round-up mode.
* - __bfloat162int_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162int_ru \cuda_math_formula (x), x > INT_MAX\end_cuda_math_formula returns INT_MAX = \p 0x7FFFFFFF.
* - __bfloat162int_ru \cuda_math_formula (x), x < INT_MIN\end_cuda_math_formula returns INT_MIN = \p 0x80000000.
* - __bfloat162int_ru(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ int __bfloat162int_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed integer to a nv_bfloat16 in round-to-nearest-even mode.
*
* \details Convert the signed integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rn(const int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed integer to a nv_bfloat16 in round-towards-zero mode.
*
* \details Convert the signed integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rz(const int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the signed integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rd(const int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the signed integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_ru(const int i);

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed short integer in round-to-nearest-even
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns short int
* - \p h converted to a signed short integer using round-to-nearest-even mode.
* - __bfloat162short_rn \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162short_rn \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __bfloat162short_rn \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __bfloat162short_rn(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ short int __bfloat162short_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed short integer in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns short int
* - \p h converted to a signed short integer using round-towards-zero mode.
* - __bfloat162short_rz \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162short_rz \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __bfloat162short_rz \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __bfloat162short_rz(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ short int __bfloat162short_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed short integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns short int
* - \p h converted to a signed short integer using round-down mode.
* - __bfloat162short_rd \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162short_rd \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __bfloat162short_rd \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __bfloat162short_rd(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ short int __bfloat162short_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed short integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns short int
* - \p h converted to a signed short integer using round-up mode.
* - __bfloat162short_ru \cuda_math_formula (\pm 0)\end_cuda_math_formula returns 0.
* - __bfloat162short_ru \cuda_math_formula (x), x > 32767\end_cuda_math_formula returns SHRT_MAX = \p 0x7FFF.
* - __bfloat162short_ru \cuda_math_formula (x), x < -32768\end_cuda_math_formula returns SHRT_MIN = \p 0x8000.
* - __bfloat162short_ru(NaN) returns 0.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ short int __bfloat162short_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed short integer to a nv_bfloat16 in round-to-nearest-even
* mode.
*
* \details Convert the signed short integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rn(const short int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed short integer to a nv_bfloat16 in round-towards-zero mode.
*
* \details Convert the signed short integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rz(const short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed short integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the signed short integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rd(const short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed short integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the signed short integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_ru(const short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned integer in round-to-nearest-even mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned integer
* in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned int
* - \p h converted to an unsigned integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned int __bfloat162uint_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned integer in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned integer
* in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned int
* - \p h converted to an unsigned integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __bfloat162uint_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned integer
* in round-down mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned int
* - \p h converted to an unsigned integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned int __bfloat162uint_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned integer
* in round-up mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned int
* - \p h converted to an unsigned integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned int __bfloat162uint_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned integer to a nv_bfloat16 in round-to-nearest-even mode.
*
* \details Convert the unsigned integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rn(const unsigned int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned integer to a nv_bfloat16 in round-towards-zero mode.
*
* \details Convert the unsigned integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rz(const unsigned int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the unsigned integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - unsigned int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rd(const unsigned int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the unsigned integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - unsigned int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_ru(const unsigned int i);

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned short integer in round-to-nearest-even
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned short int
* - \p h converted to an unsigned short integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned short integer in round-towards-zero
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned short int
* - \p h converted to an unsigned short integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat162ushort_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned short integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned short int
* - \p h converted to an unsigned short integer.
*/
__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned short integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned short int
* - \p h converted to an unsigned short integer.
*/
__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned short integer to a nv_bfloat16 in round-to-nearest-even
* mode.
*
* \details Convert the unsigned short integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rn(const unsigned short int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned short integer to a nv_bfloat16 in round-towards-zero
* mode.
*
* \details Convert the unsigned short integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rz(const unsigned short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned short integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the unsigned short integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - unsigned short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rd(const unsigned short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned short integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the unsigned short integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - unsigned short int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_ru(const unsigned short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned 64-bit integer in round-to-nearest-even
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned 64-bit
* integer in round-to-nearest-even mode. NaN inputs return 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned 64-bit integer in round-towards-zero
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned 64-bit
* integer in round-towards-zero mode. NaN inputs return 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned long long int __bfloat162ull_rz(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Vector function, combines two \p nv_bfloat16 numbers into one \p nv_bfloat162 number.
*
* \details Combines two input \p nv_bfloat16 number \p x and \p y into one \p nv_bfloat162 number.
* Input \p x is stored in low 16 bits of the return value, input \p y is stored
* in high 16 bits of the return value.
* \param[in] x - nv_bfloat16. Is only being read.
* \param[in] y - nv_bfloat16. Is only being read.
*
* \returns __nv_bfloat162
* - The \p __nv_bfloat162 vector with one half equal to \p x and the other to \p y.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned 64-bit integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned 64-bit
* integer in round-down mode. NaN inputs return 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to an unsigned 64-bit integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to an unsigned 64-bit
* integer in round-up mode. NaN inputs return 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned long long int
* - \p h converted to an unsigned 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned 64-bit integer to a nv_bfloat16 in round-to-nearest-even
* mode.
*
* \details Convert the unsigned 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rn(const unsigned long long int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned 64-bit integer to a nv_bfloat16 in round-towards-zero
* mode.
*
* \details Convert the unsigned 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rz(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned 64-bit integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the unsigned 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - unsigned long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rd(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert an unsigned 64-bit integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the unsigned 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - unsigned long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_ru(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed 64-bit integer in round-to-nearest-even
* mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed 64-bit
* integer in round-to-nearest-even mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns long long int
* - \p h converted to a signed 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ long long int __bfloat162ll_rn(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed 64-bit integer in round-towards-zero mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed 64-bit
* integer in round-towards-zero mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns long long int
* - \p h converted to a signed 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ long long int __bfloat162ll_rz(const __nv_bfloat16 h);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed 64-bit integer in round-down mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed 64-bit
* integer in round-down mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns long long int
* - \p h converted to a signed 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ long long int __bfloat162ll_rd(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a nv_bfloat16 to a signed 64-bit integer in round-up mode.
*
* \details Convert the nv_bfloat16 floating-point value \p h to a signed 64-bit
* integer in round-up mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns long long int
* - \p h converted to a signed 64-bit integer.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ long long int __bfloat162ll_ru(const __nv_bfloat16 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed 64-bit integer to a nv_bfloat16 in round-to-nearest-even
* mode.
*
* \details Convert the signed 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-to-nearest-even mode.
* \param[in] i - long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rn(const long long int i);
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed 64-bit integer to a nv_bfloat16 in round-towards-zero mode.
*
* \details Convert the signed 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-towards-zero mode.
* \param[in] i - long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rz(const long long int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed 64-bit integer to a nv_bfloat16 in round-down mode.
*
* \details Convert the signed 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-down mode.
* \param[in] i - long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rd(const long long int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Convert a signed 64-bit integer to a nv_bfloat16 in round-up mode.
*
* \details Convert the signed 64-bit integer value \p i to a nv_bfloat16 floating-point
* value in round-up mode.
* \param[in] i - long long int. Is only being read.
*
* \returns nv_bfloat16
* - \p i converted to nv_bfloat16.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_ru(const long long int i);

/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Truncate input argument to the integral part.
*
* \details Round \p h to the nearest integer value that does not exceed \p h in
* magnitude.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The truncated integer value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 htrunc(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculate ceiling of the input argument.
*
* \details Compute the smallest integer value not less than \p h.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The smallest integer value not less than \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hceil(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
*
* \details Calculate the largest integer value which is less than or equal to \p h.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The largest integer value which is less than or equal to \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hfloor(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Round input to nearest integer value in nv_bfloat16 floating-point
* number.
*
* \details Round \p h to the nearest integer value in nv_bfloat16 floating-point
* format, with bfloat16way cases rounded to the nearest even integer value.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The nearest integer to \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hrint(const __nv_bfloat16 h);

/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Truncate \p nv_bfloat162 vector input argument to the integral part.
*
* \details Round each component of vector \p h to the nearest integer value that does
* not exceed \p h in magnitude.
* \param[in] h - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The truncated \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2trunc(const __nv_bfloat162 h);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculate \p nv_bfloat162 vector ceiling of the input argument.
*
* \details For each component of vector \p h compute the smallest integer value not less
* than \p h.
* \param[in] h - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector of smallest integers not less than \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2ceil(const __nv_bfloat162 h);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
*
* \details For each component of vector \p h calculate the largest integer value which
* is less than or equal to \p h.
* \param[in] h - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector of largest integers which is less than or equal to \p h.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2floor(const __nv_bfloat162 h);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Round input to nearest integer value in nv_bfloat16 floating-point
* number.
*
* \details Round each component of \p nv_bfloat162 vector \p h to the nearest integer value in
* nv_bfloat16 floating-point format, with bfloat16way cases rounded to the
* nearest even integer value.
* \param[in] h - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector of rounded integer values.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2rint(const __nv_bfloat162 h);
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Returns \p nv_bfloat162 with both halves equal to the input value.
*
* \details Returns \p nv_bfloat162 number with both halves equal to the input \p a \p nv_bfloat16
* number.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat162
* - The vector which has both its halves equal to the input \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __bfloat162bfloat162(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Swaps both halves of the \p nv_bfloat162 input.
*
* \details Swaps both halves of the \p nv_bfloat162 input and returns a new \p nv_bfloat162 number
* with swapped halves.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - \p a with its halves being swapped.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __lowhigh2highlow(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Extracts low 16 bits from each of the two \p nv_bfloat162 inputs and combines
* into one \p nv_bfloat162 number.
*
* \details Extracts low 16 bits from each of the two \p nv_bfloat162 inputs and combines into
* one \p nv_bfloat162 number. Low 16 bits from input \p a is stored in low 16 bits of
* the return value, low 16 bits from input \p b is stored in high 16 bits of
* the return value.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The low 16 bits of \p a and of \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __lows2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Extracts high 16 bits from each of the two \p nv_bfloat162 inputs and
* combines into one \p nv_bfloat162 number.
*
* \details Extracts high 16 bits from each of the two \p nv_bfloat162 inputs and combines into
* one \p nv_bfloat162 number. High 16 bits from input \p a is stored in low 16 bits of
* the return value, high 16 bits from input \p b is stored in high 16 bits of
* the return value.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The high 16 bits of \p a and of \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __highs2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Returns high 16 bits of \p nv_bfloat162 input.
*
* \details Returns high 16 bits of \p nv_bfloat162 input \p a.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat16
* - The high 16 bits of the input.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Returns low 16 bits of \p nv_bfloat162 input.
*
* \details Returns low 16 bits of \p nv_bfloat162 input \p a.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat16
* - Returns \p nv_bfloat16 which contains low 16 bits of the input \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __low2bfloat16(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Checks if the input \p nv_bfloat16 number is infinite.
*
* \details Checks if the input \p nv_bfloat16 number \p a is infinite.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns int
* - -1 if \p a is equal to negative infinity,
* - 1 if \p a is equal to positive infinity,
* - 0 otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ int __hisinf(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Combines two \p nv_bfloat16 numbers into one \p nv_bfloat162 number.
*
* \details Combines two input \p nv_bfloat16 number \p a and \p b into one \p nv_bfloat162 number.
* Input \p a is stored in low 16 bits of the return value, input \p b is stored
* in high 16 bits of the return value.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat162
* - The nv_bfloat162 with one nv_bfloat16 equal to \p a and the other to \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __halves2bfloat162(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Extracts low 16 bits from \p nv_bfloat162 input.
*
* \details Extracts low 16 bits from \p nv_bfloat162 input \p a and returns a new \p nv_bfloat162
* number which has both halves equal to the extracted bits.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The nv_bfloat162 with both halves equal to the low 16 bits of the input.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __low2bfloat162(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Extracts high 16 bits from \p nv_bfloat162 input.
*
* \details Extracts high 16 bits from \p nv_bfloat162 input \p a and returns a new \p nv_bfloat162
* number which has both halves equal to the extracted bits.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The nv_bfloat162 with both halves equal to the high 16 bits of the input.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __high2bfloat162(const __nv_bfloat162 a);

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Reinterprets bits in a \p nv_bfloat16 as a signed short integer.
*
* \details Reinterprets the bits in the nv_bfloat16 floating-point number \p h
* as a signed short integer.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns short int
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ short int __bfloat16_as_short(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Reinterprets bits in a \p nv_bfloat16 as an unsigned short integer.
*
* \details Reinterprets the bits in the nv_bfloat16 floating-point \p h
* as an unsigned short number.
* \param[in] h - nv_bfloat16. Is only being read.
*
* \returns unsigned short int
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat16_as_ushort(const __nv_bfloat16 h);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Reinterprets bits in a signed short integer as a \p nv_bfloat16.
*
* \details Reinterprets the bits in the signed short integer \p i as a
* nv_bfloat16 floating-point number.
* \param[in] i - short int. Is only being read.
*
* \returns nv_bfloat16
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __short_as_bfloat16(const short int i);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Reinterprets bits in an unsigned short integer as a \p nv_bfloat16.
*
* \details Reinterprets the bits in the unsigned short integer \p i as a
* nv_bfloat16 floating-point number.
* \param[in] i - unsigned short int. Is only being read.
*
* \returns nv_bfloat16
* - The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ushort_as_bfloat16(const unsigned short int i);

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300))) || defined(_NVHPC_CUDA)

#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread.
*
* \details Returns the value of \p var held by the thread whose ID is given by \p srcLane.
* If the \p width is less than \p warpSize, then each subsection of the warp behaves as a separate
* entity with a starting logical thread ID of 0. If \p srcLane is outside the range \p [0:width-1],
* the value returned corresponds to the value of \p var held by the \p srcLane modulo \p width (i.e.
* within the same subsection). \p width must have a value which is a power of 2;
* results are undefined if \p width is not a power of 2, or is a number greater than
* \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat162. Is only being read.
* \param[in] srcLane - int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p nv_bfloat162.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_sync(const unsigned int mask, const __nv_bfloat162 var, const int srcLane, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller.
*
* \details Calculates a source thread ID by subtracting \p delta from the caller's lane ID.
* The value of \p var held by the resulting lane ID is returned: in effect, \p var is shifted up
* the warp by \p delta threads. If the \p width is less than \p warpSize, then each subsection of the warp
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index
* will not wrap around the value of \p width, so effectively the lower \p delta threads will be unchanged.
* \p width must have a value which is a power of 2; results are undefined if \p width is not a power of 2,
* or is a number greater than \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat162. Is only being read.
* \param[in] delta - unsigned int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p nv_bfloat162.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_up_sync(const unsigned int mask, const __nv_bfloat162 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller.
*
* \details Calculates a source thread ID by adding \p delta to the caller's thread ID.
* The value of \p var held by the resulting thread ID is returned: this has the effect
* of shifting \p var down the warp by \p delta threads. If the \p width is less than \p warpSize, then
* each subsection of the warp behaves as a separate entity with a starting logical
* thread ID of 0. Similarly to the __shfl_up_sync(), the ID number of the source thread
* will not wrap around the value of \p width and the upper \p delta threads
* will remain unchanged.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat162. Is only being read.
* \param[in] delta - unsigned int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p nv_bfloat162.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_down_sync(const unsigned int mask, const __nv_bfloat162 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID.
*
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with \p laneMask:
* the value of \p var held by the resulting thread ID is returned. If the \p width is less than \p warpSize, then each
* group of \p width consecutive threads are able to access elements from earlier groups of threads,
* however if they attempt to access elements from later groups of threads their own value of \p var
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree
* reduction and broadcast.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat162. Is only being read.
* \param[in] laneMask - int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 4-byte word referenced by \p var from the source thread ID as \p nv_bfloat162.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_xor_sync(const unsigned int mask, const __nv_bfloat162 var, const int laneMask, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread.
*
* \details Returns the value of \p var held by the thread whose ID is given by \p srcLane.
* If the \p width is less than \p warpSize, then each subsection of the warp behaves as a separate
* entity with a starting logical thread ID of 0. If \p srcLane is outside the range \p [0:width-1],
* the value returned corresponds to the value of \p var held by the \p srcLane modulo \p width (i.e.
* within the same subsection). \p width must have a value which is a power of 2;
* results are undefined if \p width is not a power of 2, or is a number greater than
* \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat16. Is only being read.
* \param[in] srcLane - int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p nv_bfloat16.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_sync(const unsigned int mask, const __nv_bfloat16 var, const int srcLane, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller.
*
* \details Calculates a source thread ID by subtracting \p delta from the caller's lane ID.
* The value of \p var held by the resulting lane ID is returned: in effect, \p var is shifted up
* the warp by \p delta threads. If the \p width is less than \p warpSize, then each subsection of the warp
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index
* will not wrap around the value of \p width, so effectively the lower \p delta threads will be unchanged.
* \p width must have a value which is a power of 2; results are undefined if \p width is not a power of 2,
* or is a number greater than \p warpSize.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat16. Is only being read.
* \param[in] delta - unsigned int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p nv_bfloat16.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_up_sync(const unsigned int mask, const __nv_bfloat16 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller.
*
* \details Calculates a source thread ID by adding \p delta to the caller's thread ID.
* The value of \p var held by the resulting thread ID is returned: this has the effect
* of shifting \p var down the warp by \p delta threads. If the \p width is less than \p warpSize, then
* each subsection of the warp behaves as a separate entity with a starting logical
* thread ID of 0. Similarly to the __shfl_up_sync(), the ID number of the source thread
* will not wrap around the value of \p width and the upper \p delta threads
* will remain unchanged.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat16. Is only being read.
* \param[in] delta - unsigned int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p nv_bfloat16.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_down_sync(const unsigned int mask, const __nv_bfloat16 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID.
*
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with \p laneMask:
* the value of \p var held by the resulting thread ID is returned. If the \p width is less than \p warpSize, then each
* group of \p width consecutive threads are able to access elements from earlier groups of threads,
* however if they attempt to access elements from later groups of threads their own value of \p var
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree
* reduction and broadcast.
* Threads may only read data from another thread which is actively participating in the
* \p __shfl_*sync() command. If the target thread is inactive, the retrieved value is undefined.
* \param[in] mask - unsigned int. Is only being read.
*  - Indicates the threads participating in the call.
*  - A bit, representing the thread's lane ID, must be set for each participating thread
*    to ensure they are properly converged before the intrinsic is executed by the hardware.
*  - Each calling thread must have its own bit set in the \p mask and all non-exited threads
*    named in \p mask must execute the same intrinsic with the same \p mask, or the result is undefined.
* \param[in] var - nv_bfloat16. Is only being read.
* \param[in] laneMask - int. Is only being read.
* \param[in] width - int. Is only being read.
*
* \returns Returns the 2-byte word referenced by \p var from the source thread ID as \p nv_bfloat16.
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_xor_sync(const unsigned int mask, const __nv_bfloat16 var, const int laneMask, const int width = warpSize);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300))) || defined(_NVHPC_CUDA) */

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320))) || defined(_NVHPC_CUDA)
#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldg(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldg(const __nv_bfloat16 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcg(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcg(const __nv_bfloat16 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldca(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldca(const __nv_bfloat16 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcs(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcs(const __nv_bfloat16 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldlu(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldlu(const __nv_bfloat16 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcv(const  __nv_bfloat162 *const ptr);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcv(const __nv_bfloat16 *const ptr);

/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stwb(__nv_bfloat162 *const ptr, const __nv_bfloat162 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stwb(__nv_bfloat16 *const ptr, const __nv_bfloat16 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stcg(__nv_bfloat162 *const ptr, const __nv_bfloat162 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stcg(__nv_bfloat16 *const ptr, const __nv_bfloat16 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stcs(__nv_bfloat162 *const ptr, const __nv_bfloat162 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stcs(__nv_bfloat16 *const ptr, const __nv_bfloat16 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stwt(__nv_bfloat162 *const ptr, const __nv_bfloat162 value);
/**
* \ingroup CUDA_MATH__BFLOAT16_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_BF16_DECL__ void __stwt(__nv_bfloat16 *const ptr, const __nv_bfloat16 value);

#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320))) || defined(_NVHPC_CUDA) */

/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs nv_bfloat162 vector if-equal comparison.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __heq2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector not-equal comparison.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hne2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-equal comparison.
*
* \details Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 result of less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hle2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-equal comparison.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hge2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-than comparison.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The nv_bfloat162 vector result of less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hlt2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-than comparison.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgt2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered if-equal comparison.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of unordered if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hequ2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered not-equal comparison.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of unordered not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hneu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-equal comparison.
*
* Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of unordered less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hleu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-equal comparison.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 vector result of unordered greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgeu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-than comparison.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The vector result of unordered less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hltu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-than comparison.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p nv_bfloat16 results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The \p nv_bfloat162 vector result of unordered greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgtu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs nv_bfloat162 vector if-equal comparison.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __heq2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector not-equal comparison.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hne2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-equal comparison.
*
* \details Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hle2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-equal comparison.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hge2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-than comparison.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hlt2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-than comparison.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hgt2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered if-equal comparison.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hequ2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered not-equal comparison.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hneu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-equal comparison.
*
* Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hleu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-equal comparison.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hgeu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-than comparison.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hltu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-than comparison.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p unsigned bits are set to 0xFFFF for true, or 0x0 for false.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns unsigned int
* - The vector mask result of unordered greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __hgtu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Determine whether \p nv_bfloat162 argument is a NaN.
*
* \details Determine whether each nv_bfloat16 of input \p nv_bfloat162 number \p a is a NaN.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The nv_bfloat162 with the corresponding \p nv_bfloat16 results set to
* 1.0 for NaN, 0.0 otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hisnan2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector addition in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat162 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The sum of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p nv_bfloat162 input vector \p b from input vector \p a in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The subtraction of vector \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat162 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise multiplying the vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector addition in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat162 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The sum of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p nv_bfloat162 input vector \p b from input vector \p a in
* round-to-nearest-even mode. Prevents floating-point contractions of mul+sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The subtraction of vector \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat162 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode. Prevents floating-point contractions of mul+add
* or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise multiplying the vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector division in round-to-nearest-even mode.
*
* \details Divides \p nv_bfloat162 input vector \p a by input vector \p b in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-103
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise division of \p a with \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __h2div(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Calculates the absolute value of both halves of the input \p nv_bfloat162 number and
* returns the result.
*
* \details Calculates the absolute value of both halves of the input \p nv_bfloat162 number and
* returns the result.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns bfloat2
* - Returns \p a with the absolute value of both halves.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __habs2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat162 vector add of inputs \p a and \p b, in round-to-nearest-even
* mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The sum of \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector subtraction in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Subtracts \p nv_bfloat162 input vector \p b from input vector \p a in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The subtraction of vector \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector multiplication in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat162 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise multiplication of vectors \p a and \p b,
* with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b);
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector fused multiply-add in round-to-nearest-even
* mode.
*
* \details Performs \p nv_bfloat162 vector multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat162 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-105
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
* \param[in] c - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector fused multiply-add in round-to-nearest-even
* mode, with saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat162 vector multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat162 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the
* results to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
* \param[in] c - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c,
* with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Negates both halves of the input \p nv_bfloat162 number and returns the
* result.
*
* \details Negates both halves of the input \p nv_bfloat162 number \p a and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-101
* \endinternal
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - Returns \p a with both halves negated.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hneg2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Calculates the absolute value of input \p nv_bfloat16 number and returns the result.
*
* \details Calculates the absolute value of input \p nv_bfloat16 number and returns the result.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The absolute value of a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __habs(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 addition in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat16 addition of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The sum of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p nv_bfloat16 input \p b from input \p a in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of subtracting \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 multiplication in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat16 multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of multiplying \p a and \p b.
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 addition in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat16 addition of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The sum of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd_rn(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p nv_bfloat16 input \p b from input \p a in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of subtracting \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub_rn(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 multiplication in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat16 multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of multiplying \p a and \p b.
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul_rn(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 division in round-to-nearest-even mode.
*
* \details Divides \p nv_bfloat16 input \p a by input \p b in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-98
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of dividing \p a by \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__  __nv_bfloat16 __hdiv(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat16 add of inputs \p a and \p b, in round-to-nearest-even mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The sum of \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd_sat(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 subtraction in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Subtracts \p nv_bfloat16 input \p b from input \p a in round-to-nearest-even
* mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of subtraction of \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub_sat(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 multiplication in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat16 multiplication of inputs \p a and \p b, in round-to-nearest-even
* mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of multiplying \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul_sat(const __nv_bfloat16 a, const __nv_bfloat16 b);
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 fused multiply-add in round-to-nearest-even mode.
*
* \details Performs \p nv_bfloat16 multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat16 add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-96
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
* \param[in] c - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c);
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 fused multiply-add in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p nv_bfloat16 multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat16 add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the result
* to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
* \param[in] c - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma_sat(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Negates input \p nv_bfloat16 number and returns the result.
*
* \details Negates input \p nv_bfloat16 number and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-100
* \endinternal
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - minus a
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hneg(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector if-equal comparison and returns boolean true
* if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of if-equal comparison
* of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbeq2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector not-equal comparison and returns boolean
* true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of not-equal comparison
* of vectors \p a and \p b are true,
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbne2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-equal comparison and returns boolean
* true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of less-equal comparison
* of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hble2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-equal comparison and returns boolean
* true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of greater-equal
* comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbge2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector less-than comparison and returns boolean
* true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of less-than comparison
* of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hblt2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector greater-than comparison and returns boolean
* true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of greater-than
* comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgt2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered if-equal comparison and returns
* boolean true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered if-equal
* comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbequ2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered not-equal comparison and returns
* boolean true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered not-equal
* comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbneu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-equal comparison and returns
* boolean true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered less-equal
* comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbleu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-equal comparison and
* returns boolean true if both \p nv_bfloat16 results are true, boolean false
* otherwise.
*
* \details Performs \p nv_bfloat162 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered
* greater-equal comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgeu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered less-than comparison and returns
* boolean true if both \p nv_bfloat16 results are true, boolean false otherwise.
*
* \details Performs \p nv_bfloat162 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered less-than comparison of
* vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbltu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Performs \p nv_bfloat162 vector unordered greater-than comparison and
* returns boolean true if both \p nv_bfloat16 results are true, boolean false
* otherwise.
*
* \details Performs \p nv_bfloat162 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p nv_bfloat16 greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns bool
* - true if both \p nv_bfloat16 results of unordered
* greater-than comparison of vectors \p a and \p b are true;
* - false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgtu2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 if-equal comparison.
*
* \details Performs \p nv_bfloat16 if-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of if-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __heq(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 not-equal comparison.
*
* \details Performs \p nv_bfloat16 not-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of not-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hne(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 less-equal comparison.
*
* \details Performs \p nv_bfloat16 less-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of less-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hle(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 greater-equal comparison.
*
* \details Performs \p nv_bfloat16 greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of greater-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hge(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 less-than comparison.
*
* \details Performs \p nv_bfloat16 less-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of less-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hlt(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 greater-than comparison.
*
* \details Performs \p nv_bfloat16 greater-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of greater-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgt(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered if-equal comparison.
*
* \details Performs \p nv_bfloat16 if-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered if-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hequ(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered not-equal comparison.
*
* \details Performs \p nv_bfloat16 not-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered not-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hneu(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered less-equal comparison.
*
* \details Performs \p nv_bfloat16 less-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered less-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hleu(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered greater-equal comparison.
*
* \details Performs \p nv_bfloat16 greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered greater-equal comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgeu(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered less-than comparison.
*
* \details Performs \p nv_bfloat16 less-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered less-than comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hltu(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Performs \p nv_bfloat16 unordered greater-than comparison.
*
* \details Performs \p nv_bfloat16 greater-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns bool
* - The boolean result of unordered greater-than comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgtu(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Determine whether \p nv_bfloat16 argument is a NaN.
*
* \details Determine whether \p nv_bfloat16 value \p a is a NaN.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns bool
* - true if argument is NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hisnan(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Calculates \p nv_bfloat16 maximum of two input values.
*
* \details Calculates \p nv_bfloat16 max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmax(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Calculates \p nv_bfloat16 minimum of two input values.
*
* \details Calculates \p nv_bfloat16 min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmin(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Calculates \p nv_bfloat16 maximum of two input values, NaNs pass through.
*
* \details Calculates \p nv_bfloat16 max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmax_nan(const __nv_bfloat16 a, const __nv_bfloat16 b);
/**
* \ingroup CUDA_MATH__BFLOAT16_COMPARISON
* \brief Calculates \p nv_bfloat16 minimum of two input values, NaNs pass through.
*
* \details Calculates \p nv_bfloat16 min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmin_nan(const __nv_bfloat16 a, const __nv_bfloat16 b);
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Performs \p nv_bfloat16 fused multiply-add in round-to-nearest-even mode with relu saturation.
*
* \details Performs \p nv_bfloat16 multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat16 add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - nv_bfloat16. Is only being read.
* \param[in] b - nv_bfloat16. Is only being read.
* \param[in] c - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The result of fused multiply-add operation on \p
* a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma_relu(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Calculates \p nv_bfloat162 vector maximum of two inputs.
*
* \details Calculates \p nv_bfloat162 vector max(\p a, \p b).
* Elementwise \p nv_bfloat16 operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise maximum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmax2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Calculates \p nv_bfloat162 vector minimum of two inputs.
*
* \details Calculates \p nv_bfloat162 vector min(\p a, \p b).
* Elementwise \p nv_bfloat16 operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise minimum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmin2(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Calculates \p nv_bfloat162 vector maximum of two inputs, NaNs pass through.
*
* \details Calculates \p nv_bfloat162 vector max(\p a, \p b).
* Elementwise \p nv_bfloat16 operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise maximum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmax2_nan(const __nv_bfloat162 a, const __nv_bfloat162 b);
/**
* \ingroup CUDA_MATH__BFLOAT162_COMPARISON
* \brief Calculates \p nv_bfloat162 vector minimum of two inputs, NaNs pass through.
*
* \details Calculates \p nv_bfloat162 vector min(\p a, \p b).
* Elementwise \p nv_bfloat16 operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise minimum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmin2_nan(const __nv_bfloat162 a, const __nv_bfloat162 b);
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs \p nv_bfloat162 vector fused multiply-add in round-to-nearest-even
* mode with relu saturation.
*
* \details Performs \p nv_bfloat162 vector multiply on inputs \p a and \p b,
* then performs a \p nv_bfloat162 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
* \param[in] c - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2_relu(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c);
/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Performs fast complex multiply-accumulate
*
* \details Interprets vector \p nv_bfloat162 input pairs \p a, \p b, and \p c as
* complex numbers in \p nv_bfloat16 precision and performs
* complex multiply-accumulate operation: a*b + c
* \param[in] a - nv_bfloat162. Is only being read.
* \param[in] b - nv_bfloat162. Is only being read.
* \param[in] c - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The result of complex multiply-accumulate operation on complex numbers \p a, \p b, and \p c
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 __hcmadd(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 square root in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 square root of input \p a in round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The square root of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hsqrt(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 reciprocal square root in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat16 reciprocal square root of input \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The reciprocal square root of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hrsqrt(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 reciprocal of input \p a in round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The reciprocal of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hrcp(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 natural logarithm in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 natural logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The natural logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hlog(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 binary logarithm in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 binary logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The binary logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hlog2(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 decimal logarithm in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 decimal logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The decimal logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hlog10(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 natural exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat16 natural exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The natural exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hexp(const __nv_bfloat16 a);

/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates approximate \p nv_bfloat16 hyperbolic tangent function.
*
* \details Calculates approximate \p nv_bfloat16 hyperbolic tangent function: \cuda_math_formula \tanh(a)\end_cuda_math_formula.
* This operation uses HW acceleration on devices of compute capability 9.x and higher.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The approximate hyperbolic tangent function of \p a.
* - htanh_approx \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula (\pm 0)\end_cuda_math_formula.
* - htanh_approx \cuda_math_formula (\pm\infty)\end_cuda_math_formula returns \cuda_math_formula (\pm 1)\end_cuda_math_formula.
* - htanh_approx(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 htanh_approx(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector approximate hyperbolic tangent function.
*
* \details Calculates \p nv_bfloat162 approximate hyperbolic tangent function of input vector \p a.
* This operation uses HW acceleration on devices of compute capability 9.x and higher.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise approximate hyperbolic tangent function on vector \p a.
*
* \see htanh_approx(__nv_bfloat16) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2tanh_approx(const __nv_bfloat162 a);

/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 hyperbolic tangent function in
* round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 hyperbolic tangent function: \cuda_math_formula \tanh(a)\end_cuda_math_formula in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The hyperbolic tangent function of \p a.
* - htanh \cuda_math_formula (\pm 0)\end_cuda_math_formula returns \cuda_math_formula (\pm 0)\end_cuda_math_formula.
* - htanh \cuda_math_formula (\pm\infty)\end_cuda_math_formula returns \cuda_math_formula (\pm 1)\end_cuda_math_formula.
* - htanh(NaN) returns NaN.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 htanh(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector hyperbolic tangent function in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 hyperbolic tangent function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise hyperbolic tangent function on vector \p a.
*
* \see htanh(__nv_bfloat16) for further details.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2tanh(const __nv_bfloat162 a);

/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 binary exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat16 binary exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The binary exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hexp2(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 decimal exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat16 decimal exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The decimal exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hexp10(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 cosine in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 cosine of input \p a in round-to-nearest-even mode.
*
* NOTE: this function's implementation calls cosf(float) function and is exposed
* to compiler optimizations. Specifically, \p --use_fast_math flag changes cosf(float)
* into an intrinsic __cosf(float), which has less accurate numeric behavior.
*
* \param[in] a - nv_bfloat16. Is only being read.
* \returns nv_bfloat16
* - The cosine of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hcos(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT16_FUNCTIONS
* \brief Calculates \p nv_bfloat16 sine in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat16 sine of input \p a in round-to-nearest-even mode.
*
* NOTE: this function's implementation calls sinf(float) function and is exposed
* to compiler optimizations. Specifically, \p --use_fast_math flag changes sinf(float)
* into an intrinsic __sinf(float), which has less accurate numeric behavior.
*
* \param[in] a - nv_bfloat16. Is only being read.
*
* \returns nv_bfloat16
* - The sine of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat16 hsin(const __nv_bfloat16 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector square root in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 square root of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise square root on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2sqrt(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector reciprocal square root in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 reciprocal square root of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise reciprocal square root on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2rsqrt(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 reciprocal of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise reciprocal on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2rcp(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector natural logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 natural logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise natural logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2log(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector binary logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 binary logarithm of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise binary logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2log2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector decimal logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 decimal logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise decimal logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2log10(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector exponential function in round-to-nearest-even
* mode.
*
* \details Calculates \p nv_bfloat162 exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2exp(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector binary exponential function in
* round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 binary exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise binary exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2exp2(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector decimal exponential function in
* round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 decimal exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - nv_bfloat162. Is only being read.
*
* \returns nv_bfloat162
* - The elementwise decimal exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2exp10(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector cosine in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 cosine of input vector \p a in round-to-nearest-even
* mode.
*
* NOTE: this function's implementation calls cosf(float) function and is exposed
* to compiler optimizations. Specifically, \p --use_fast_math flag changes cosf(float)
* into an intrinsic __cosf(float), which has less accurate numeric behavior.
*
* \param[in] a - nv_bfloat162. Is only being read.
* \returns nv_bfloat162
* - The elementwise cosine on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2cos(const __nv_bfloat162 a);
/**
* \ingroup CUDA_MATH__BFLOAT162_FUNCTIONS
* \brief Calculates \p nv_bfloat162 vector sine in round-to-nearest-even mode.
*
* \details Calculates \p nv_bfloat162 sine of input vector \p a in round-to-nearest-even mode.
*
* NOTE: this function's implementation calls sinf(float) function and is exposed
* to compiler optimizations. Specifically, \p --use_fast_math flag changes sinf(float)
* into an intrinsic __sinf(float), which has less accurate numeric behavior.
*
* \param[in] a - nv_bfloat162. Is only being read.
* \returns nv_bfloat162
* - The elementwise sine on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_BF16_DECL__ __nv_bfloat162 h2sin(const __nv_bfloat162 a);

/**
* \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
* \brief Vector add \p val to the value stored at \p address in global or shared memory, and writes this
* value back to \p address. The atomicity of the add operation is guaranteed separately for each of the
* two nv_bfloat16 elements; the entire __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.
*
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is natively supported by devices of compute capability 9.x and higher,
* older devices use emulation path.
*
* \param[in] address - __nv_bfloat162*. An address in global or shared memory.
* \param[in] val - __nv_bfloat162. The value to be added.
*
* \returns __nv_bfloat162
* - The old value read from \p address.
*
* \note_ref_guide_atomic
*/
__CUDA_BF16_DECL__ __nv_bfloat162 atomicAdd(__nv_bfloat162 *const address, const __nv_bfloat162 val);

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
/**
* \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
* \brief Adds \p val to the value stored at \p address in global or shared memory, and writes this value
* back to \p address. This operation is performed in one atomic operation.
*
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is natively supported by devices of compute capability 9.x and higher,
* older devices of compute capability 7.x and 8.x use emulation path.
*
* \param[in] address - __nv_bfloat16*. An address in global or shared memory.
* \param[in] val - __nv_bfloat16. The value to be added.
*
* \returns __nv_bfloat16
* - The old value read from \p address.
*
* \note_ref_guide_atomic
*/
__CUDA_BF16_DECL__ __nv_bfloat16 atomicAdd(__nv_bfloat16 *const address, const __nv_bfloat16 val);
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA) */
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */


#endif /* defined(__cplusplus) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#endif

/* C++11 header for std::move.
 * In RTC mode, std::move is provided implicitly; don't include the header
 */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__)
#include <utility>
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__) */

/* C++ header for std::memcpy (used for type punning in host-side implementations).
 * When compiling as a CUDA source file memcpy is provided implicitly.
 * !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstring>
#endif /* defined(__cplusplus) && !defined(__CUDACC__) */

// implicitly provided by NVRTC
#if !defined(__CUDACC_RTC__)
#include <nv/target>
#endif  /* !defined(__CUDACC_RTC__) */

#if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))
#define __CUDA_BF16_INLINE__
#define __CUDA_BF16_FORCEINLINE__
#else
#define __CUDA_BF16_INLINE__ inline
#define __CUDA_BF16_FORCEINLINE__ __forceinline__
#endif /* #if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))) */

/* Set up structure-alignment attribute */
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* defined(__CPP_VERSION_AT_LEAST_11_BF16)*/
#if defined(__GNUC__)
#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */
#endif /* defined(__CUDACC__) */

// define __CUDA_BF16_CONSTEXPR__ in order to
// use constexpr where possible, with supporting C++ dialects
// undef after use
#if (defined __CPP_VERSION_AT_LEAST_11_BF16)
#define __CUDA_BF16_CONSTEXPR__   constexpr
#else
#define __CUDA_BF16_CONSTEXPR__
#endif

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief __nv_bfloat16_raw data type
 * \details Type allows static initialization of \p nv_bfloat16 until it becomes
 * a built-in type.
 *
 * - Note: this initialization is as a bit-field representation of \p nv_bfloat16,
 * and not a conversion from \p short to \p nv_bfloat16.
 * Such representation will be deprecated in a future version of CUDA.
 *
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(2) {
    /**
     * Storage field contains bits representation of the \p nv_bfloat16 floating-point number.
     */
    unsigned short x;
} __nv_bfloat16_raw;

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief __nv_bfloat162_raw data type
 * \details Type allows static initialization of \p nv_bfloat162 until it becomes
 * a built-in type.
 *
 * - Note: this initialization is as a bit-field representation of \p nv_bfloat162,
 * and not a conversion from \p short2 to \p nv_bfloat162.
 * Such representation will be deprecated in a future version of CUDA.
 *
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(4) {
    /**
     * Storage field contains bits of the lower \p nv_bfloat16 part.
     */
    unsigned short x;
    /**
     * Storage field contains bits of the upper \p nv_bfloat16 part.
     */
    unsigned short y;
} __nv_bfloat162_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(__GNUC__) */

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief nv_bfloat16 datatype
 *
 * \details This structure implements the datatype for storing
 * nv_bfloat16 floating-point numbers. The structure implements
 * assignment operators and type conversions. 16 bits are being
 * used in total: 1 sign bit, 8 bits for the exponent, and
 * the significand is being stored in 7 bits. The total
 * precision is 8 bits.
 *
 */
struct __CUDA_ALIGN__(2) __nv_bfloat16 {
protected:
    /**
     * Protected storage variable contains the bits of floating-point data.
     */
    unsigned short __x;

public:

    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * \brief Constructor by default.
     * \details Emtpy default constructor, result is uninitialized.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __nv_bfloat16() = default;
#else
    __CUDA_HOSTDEVICE__ __nv_bfloat16() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Convert to/from __nv_bfloat16_raw */
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Constructor from \p __nv_bfloat16_raw.
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ __nv_bfloat16(const __nv_bfloat16_raw &hr) : __x(hr.x) { }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Assignment operator from \p __nv_bfloat16_raw.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Assignment operator from \p __nv_bfloat16_raw to \p volatile \p __nv_bfloat16.
     */
    __CUDA_HOSTDEVICE__ volatile __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr) volatile;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Assignment operator from \p volatile \p __nv_bfloat16_raw to \p volatile \p __nv_bfloat16.
     */
    __CUDA_HOSTDEVICE__ volatile __nv_bfloat16 &operator=(const volatile __nv_bfloat16_raw &hr) volatile;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast to \p __nv_bfloat16_raw operator.
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat16_raw() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast to \p __nv_bfloat16_raw operator with \p volatile input.
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat16_raw() const volatile;

#if !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__)
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p __half input using default round-to-nearest-even rounding mode.
     */
    explicit __CUDA_HOSTDEVICE__ __nv_bfloat16(const __half f)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rn.bf16.f16 %0, %1;}\n" : "=h"(__x) : "h"(__BFLOAT16_TO_CUS(f)));
,
    __x = __float2bfloat16(__half2float(f)).__x;
)
}
#endif /* #if defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Construct from float/double */
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p float input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const float f) { __x = __float2bfloat16(f).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p double input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const double f) { __x = __double2bfloat16(f).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast to \p float operator.
     */
    __CUDA_HOSTDEVICE__ operator float() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast to \p __nv_bfloat16 assignment operator from \p float input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const float f);

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast to \p __nv_bfloat16 assignment operator from \p double input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const double f);

/*
 * Implicit type conversions to/from integer types were only available to nvcc compilation.
 * Introducing them for all compilers is a potentially breaking change that may affect
 * overloads resolution and will require users to update their code.
 * Define __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ to opt-out.
 */
#if !(defined __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__)
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */

    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(short val) { __x = __short2bfloat16_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p unsigned \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned short val) { __x = __ushort2bfloat16_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(int val) { __x = __int2bfloat16_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p unsigned \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned int val) { __x = __uint2bfloat16_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const long val) {
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            __x = __ll2bfloat16_rn(static_cast<long long>(val)).__x;
        } else {
            __x = __int2bfloat16_rn(static_cast<int>(val)).__x;
        }
    }

    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p unsigned \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const unsigned long val) {
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            __x = __ull2bfloat16_rn(static_cast<unsigned long long>(val)).__x;
        } else {
            __x = __uint2bfloat16_rn(static_cast<unsigned int>(val)).__x;
        }
    }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(long long val) { __x = __ll2bfloat16_rn(val).__x; }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Construct \p __nv_bfloat16 from \p unsigned \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; }

    /* Allow automatic casts to supported built-in types, matching all that are permitted with float */

    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p signed \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162char_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator signed char() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p unsigned \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162uchar_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned char() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to an implementation defined \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.
     */
    __CUDA_HOSTDEVICE__ operator char() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p short data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162short_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator short() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p unsigned \p short data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162ushort_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned short() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p int data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162int_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator int() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p unsigned \p int data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162uint_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned int() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __CUDA_HOSTDEVICE__ operator long() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p unsigned \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __CUDA_HOSTDEVICE__ operator unsigned long() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p long \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162ll_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator long long() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p unsigned \p long \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * See __bfloat162ull_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned long long() const;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(short val);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p unsigned \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned short val);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(int val);
   /**
    * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p unsigned \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned int val);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(long long val);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Type cast from \p unsigned \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned long long val);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ operator bool() const { return (__x & 0x7FFFU) != 0U; }
#endif /* !(defined __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__) */
#endif /* !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) */
};

#if !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
/* Some basic arithmetic operations expected of a built-in */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 addition operation.
 * See also __hadd(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator+(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 subtraction operation.
 * See also __hsub(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator-(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 multiplication operation.
 * See also __hmul(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator*(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 division operation.
 * See also __hdiv(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator/(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);

/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with addition operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator+=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with subtraction operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator-=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with multiplication operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator*=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with division operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator/=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh);

/* Note for increment and decrement we use the raw value 0x3F80U equating to nv_bfloat16(1.0F), to avoid the extra conversion */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 prefix increment operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator++(__nv_bfloat16 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 prefix decrement operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 &operator--(__nv_bfloat16 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 postfix increment operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16  operator++(__nv_bfloat16 &h, const int ignored);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 postfix decrement operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16  operator--(__nv_bfloat16 &h, const int ignored);
/* Unary plus and inverse operators */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Implements \p nv_bfloat16 unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator+(const __nv_bfloat16 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Implements \p nv_bfloat16 unary minus operator.
 * See also __hneg(__nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat16 operator-(const __nv_bfloat16 &h);

/* Some basic comparison operations to make it look like a built-in */
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered compare equal operation.
 * See also __heq(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 unordered compare not-equal operation.
 * See also __hneu(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered greater-than compare operation.
 * See also __hgt(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator> (const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered less-than compare operation.
 * See also __hlt(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator< (const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered greater-or-equal compare operation.
 * See also __hge(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered less-or-equal compare operation.
 * See also __hle(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh);
#endif /* !defined(__CUDA_NO_BFLOAT16_OPERATORS__) */

/**
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief nv_bfloat162 datatype
 * \details This structure implements the datatype for storing two
 * nv_bfloat16 floating-point numbers.
 * The structure implements assignment, arithmetic and comparison
 * operators, and type conversions.
 *
 * - NOTE: __nv_bfloat162 is visible to non-nvcc host compilers
 */
struct __CUDA_ALIGN__(4) __nv_bfloat162 {
    /**
     * Storage field holding lower \p __nv_bfloat16 part.
     */
    __nv_bfloat16 x;
    /**
     * Storage field holding upper \p __nv_bfloat16 part.
     */
    __nv_bfloat16 y;

    // All construct/copy/assign/move
public:
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * \brief Constructor by default.
     * \details Emtpy default constructor, result is uninitialized.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __nv_bfloat162() = default;
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Move constructor, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(__nv_bfloat162 &&src);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Move assignment operator, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(__nv_bfloat162 &&src);
#else
    __CUDA_HOSTDEVICE__ __nv_bfloat162();
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Constructor from two \p __nv_bfloat16 variables
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ __nv_bfloat162(const __nv_bfloat16 &a, const __nv_bfloat16 &b) : x(a), y(b) { }
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Copy constructor
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(const __nv_bfloat162 &src);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Copy assignment operator
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(const __nv_bfloat162 &src);

    /* Convert to/from __nv_bfloat162_raw */
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Constructor from \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(const __nv_bfloat162_raw &h2r );
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Assignment operator from \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(const __nv_bfloat162_raw &h2r);
    /**
     * \ingroup CUDA_MATH__BFLOAT16_MISC
     * Conversion operator to \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat162_raw() const;
};

#if !defined(__CUDA_NO_BFLOAT162_OPERATORS__)
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 addition operation.
 * See also __hadd2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator+(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 subtraction operation.
 * See also __hsub2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator-(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 multiplication operation.
 * See also __hmul2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator*(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 division operation.
 * See also __h2div(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator/(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with addition operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162& operator+=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with subtraction operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162& operator-=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with multiplication operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162& operator*=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with division operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162& operator/=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 prefix increment operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 &operator++(__nv_bfloat162 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 prefix decrement operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 &operator--(__nv_bfloat162 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 postfix increment operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162  operator++(__nv_bfloat162 &h, const int ignored);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 postfix decrement operation.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162  operator--(__nv_bfloat162 &h, const int ignored);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Implements packed \p nv_bfloat16 unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator+(const __nv_bfloat162 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Implements packed \p nv_bfloat16 unary minus operator.
 * See also __hneg2(__nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ __nv_bfloat162 operator-(const __nv_bfloat162 &h);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered compare equal operation.
 * See also __hbeq2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 unordered compare not-equal operation.
 * See also __hbneu2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered greater-than compare operation.
 * See also __hbgt2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator>(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered less-than compare operation.
 * See also __hblt2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator<(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered greater-or-equal compare operation.
 * See also __hbge2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered less-or-equal compare operation.
 * See also __hble2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh);

#endif /* !defined(__CUDA_NO_BFLOAT162_OPERATORS__) */

#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
__CUDA_HOSTDEVICE__
#ifdef __CUDACC_RTC__
inline
#else
__CUDA_BF16_FORCEINLINE__
#endif
__half::__half(const __nv_bfloat16 f)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rn.f16.bf16 %0, %1;}\n" : "=h"(__x) : "h"(__BFLOAT16_TO_CUS(f)));
,
    __x = __float2half_rn(__bfloat162float(f)).__x;
)
}
#endif
#endif /* #if defined(__CPP_VERSION_AT_LEAST_11_BF16) */

#endif /* defined(__cplusplus) */

#if (defined(__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__) || \
    !(defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))))
/* Note the .hpp file is included to capture the "nv_bfloat16" & "nv_bfloat162" built-in function definitions. For NVRTC, the built-in
   function definitions are compiled at NVRTC library build-time and are available through the NVRTC built-ins library at
   link time.
*/
#include "cuda_bf16.hpp"
#endif /* (defined(__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__) || \
          !(defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))) */

/* Define first-class types "nv_bfloat16" and "nv_bfloat162", unless user specifies otherwise via "#define CUDA_NO_BFLOAT16" */
/* C cannot ever have these types defined here, because __nv_bfloat16 and __nv_bfloat162 are C++ classes */
#if defined(__cplusplus) && !defined(CUDA_NO_BFLOAT16)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of the bfloat16 numbers format.
 *
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __nv_bfloat16  nv_bfloat16;

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of type for pairs of bfloat16 numbers.
 *
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __nv_bfloat162 nv_bfloat162;

#endif /* defined(__cplusplus) && !defined(CUDA_NO_BFLOAT16) */

#undef __CUDA_BF16_DECL__
#undef __CUDA_HOSTDEVICE_BF16_DECL__
#undef __CUDA_HOSTDEVICE__
#undef __CUDA_BF16_INLINE__
#undef __CUDA_BF16_FORCEINLINE__
#undef ___CUDA_BF16_STRINGIFY_INNERMOST
#undef __CUDA_BF16_STRINGIFY

#endif /* end of include guard: __CUDA_BF16_H__ */
