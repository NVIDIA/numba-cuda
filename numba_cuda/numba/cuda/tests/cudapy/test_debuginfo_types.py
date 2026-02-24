# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import llvmlite
from numba.cuda import types

"""
llvmlite pre 45 left redundant metadata nodes for debug info
for basic scalar types. Jie Li fixed this in llvmlite 45:
https://github.com/numba/llvmlite/pull/1165

This makes it difficult for us to use the same check patterns for both
llvmlite versions, as basic scalar types like int8 are emitted only once
per module. Some checks share most of the logic, but some are entirely
split between the two versions. The checks indicate which version of llvmlite
they are for, if they cannot be shared.
"""

scalar_types: tuple[tuple[types.Type, str]] = (
    (
        types.float32,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float32", size: 32)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.float64,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float64", size: 64)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.int8,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int8", size: 8)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.int16,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int16", size: 16)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.int32,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int32", size: 32)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.int64,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.uint8,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint8", size: 8)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.uint16,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint16", size: 16)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.uint32,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint32", size: 32)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.uint64,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK: [[DBG86:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint64", size: 64)
        CHECK: !DILocalVariable(
        CHECK-SAME: name: "a"
        CHECK-SAME: type: [[DBG86]]
        """,
    ),
    (
        types.complex64,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK-LLVMLITE-LE44: [[DBG98:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
        CHECK-LLVMLITE-LE44: [[DBG99:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG98]]
        CHECK-LLVMLITE-LE44-SAME: name: "real"
        CHECK-LLVMLITE-LE44-SAME: offset: 0
        CHECK-LLVMLITE-LE44-SAME: size: 32
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG100:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
        CHECK-LLVMLITE-LE44: [[DBG101:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG100]]
        CHECK-LLVMLITE-LE44-SAME: name: "imag"
        CHECK-LLVMLITE-LE44-SAME: offset: 32
        CHECK-LLVMLITE-LE44-SAME: size: 32
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG102:.+]] = !{ [[DBG99]], [[DBG101]] }
        CHECK-LLVMLITE-LE44: [[DBG103:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-LE44-SAME: elements: [[DBG102]]
        CHECK-LLVMLITE-LE44-SAME: identifier: "{float, float}"
        CHECK-LLVMLITE-LE44-SAME: name: "complex64 ({float, float})"
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: !DILocalVariable(
        CHECK-LLVMLITE-LE44-SAME: name: "a"
        CHECK-LLVMLITE-LE44-SAME: type: [[DBG103]]
        CHECK-LLVMLITE-LE44-SAME: )

        CHECK-LLVMLITE-GE45: [[DBG89:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
        CHECK-LLVMLITE-GE45: distinct !DISubprogram
        CHECK-LLVMLITE-GE45: [[DBG97:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG89]]
        CHECK-LLVMLITE-GE45-SAME: name: "real"
        CHECK-LLVMLITE-GE45-SAME: offset: 0
        CHECK-LLVMLITE-GE45-SAME: size: 32
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG98:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG89]]
        CHECK-LLVMLITE-GE45-SAME: name: "imag"
        CHECK-LLVMLITE-GE45-SAME: offset: 32
        CHECK-LLVMLITE-GE45-SAME: size: 32
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG99:.+]] = !{ [[DBG97]], [[DBG98]] }
        CHECK-LLVMLITE-GE45: [[DBG100:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-GE45-SAME: elements: [[DBG99]]
        CHECK-LLVMLITE-GE45-SAME: identifier: "{float, float}"
        CHECK-LLVMLITE-GE45-SAME: name: "complex64 ({float, float})"
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: !DILocalVariable(
        CHECK-LLVMLITE-GE45-SAME: name: "a"
        CHECK-LLVMLITE-GE45-SAME: type: [[DBG100]]
        CHECK-LLVMLITE-GE45-SAME: )
        """,
    ),
    (
        types.complex128,
        """
        CHECK-LLVMLITE-LE44: distinct !DISubprogram
        CHECK-LLVMLITE-LE44: [[DBG98:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG99:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG98]]
        CHECK-LLVMLITE-LE44-SAME: name: "real"
        CHECK-LLVMLITE-LE44-SAME: offset: 0
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG100:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG101:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG100]]
        CHECK-LLVMLITE-LE44-SAME: name: "imag"
        CHECK-LLVMLITE-LE44-SAME: offset: 64
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG102:.+]] = !{ [[DBG99]], [[DBG101]] }
        CHECK-LLVMLITE-LE44: [[DBG103:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-LE44-SAME: elements: [[DBG102]]
        CHECK-LLVMLITE-LE44-SAME: identifier: "{double, double}"
        CHECK-LLVMLITE-LE44-SAME: name: "complex128 ({double, double})"
        CHECK-LLVMLITE-LE44-SAME: size: 128
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: !DILocalVariable(
        CHECK-LLVMLITE-LE44-SAME: name: "a"
        CHECK-LLVMLITE-LE44-SAME: type: [[DBG103]]
        CHECK-LLVMLITE-LE44-SAME: )

        CHECK-LLVMLITE-GE45: [[DBG89:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
        CHECK-LLVMLITE-GE45: !DISubprogram
        CHECK-LLVMLITE-GE45: [[DBG97:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG89]]
        CHECK-LLVMLITE-GE45-SAME: name: "real"
        CHECK-LLVMLITE-GE45-SAME: offset: 0
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG98:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG89]]
        CHECK-LLVMLITE-GE45-SAME: name: "imag"
        CHECK-LLVMLITE-GE45-SAME: offset: 64
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG99:.+]] = !{ [[DBG97]], [[DBG98]] }
        CHECK-LLVMLITE-GE45: [[DBG100:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-GE45-SAME: elements: [[DBG99]]
        CHECK-LLVMLITE-GE45-SAME: identifier: "{double, double}"
        CHECK-LLVMLITE-GE45-SAME: name: "complex128 ({double, double})"
        CHECK-LLVMLITE-GE45-SAME: size: 128
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: !DILocalVariable(
        CHECK-LLVMLITE-GE45-SAME: name: "a"
        CHECK-LLVMLITE-GE45-SAME: type: [[DBG100]]
        CHECK-LLVMLITE-GE45-SAME: )
        """,
    ),
)

array_types: tuple[tuple[types.Type, str]] = (
    (
        types.float32[::1],
        """
        CHECK: distinct !DICompileUnit
        CHECK: distinct !DISubprogram

        CHECK-LLVMLITE-LE44: [[DBG127:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i8", size: 8)
        CHECK-LLVMLITE-LE44: [[DBG128:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG127]]
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG129:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG128]]
        CHECK-LLVMLITE-LE44-SAME: name: "meminfo"
        CHECK-LLVMLITE-LE44-SAME: offset: 0
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG130:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i8", size: 8)
        CHECK-LLVMLITE-LE44: [[DBG131:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG130]]
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG132:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG131]]
        CHECK-LLVMLITE-LE44-SAME: name: "parent"
        CHECK-LLVMLITE-LE44-SAME: offset: 64
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG133:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG134:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG133]]
        CHECK-LLVMLITE-LE44-SAME: name: "nitems"
        CHECK-LLVMLITE-LE44-SAME: offset: 128
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG135:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG136:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG135]]
        CHECK-LLVMLITE-LE44-SAME: name: "itemsize"
        CHECK-LLVMLITE-LE44-SAME: offset: 192
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG137:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float32", size: 32)
        CHECK-LLVMLITE-LE44: [[DBG138:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG137]]
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG139:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG138]]
        CHECK-LLVMLITE-LE44-SAME: name: "data"
        CHECK-LLVMLITE-LE44-SAME: offset: 256
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG140:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i64", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG141:.+]] = !DICompositeType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG140]]
        CHECK-LLVMLITE-LE44-SAME: identifier: "[1 x i64]"
        CHECK-LLVMLITE-LE44-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_array_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG142:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG141]]
        CHECK-LLVMLITE-LE44-SAME: name: "shape"
        CHECK-LLVMLITE-LE44-SAME: offset: 320
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG143:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i64", size: 64)
        CHECK-LLVMLITE-LE44: [[DBG144:.+]] = !DICompositeType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG143]]
        CHECK-LLVMLITE-LE44-SAME: identifier: "[1 x i64]"
        CHECK-LLVMLITE-LE44-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_array_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG145:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-LE44-SAME: baseType: [[DBG144]]
        CHECK-LLVMLITE-LE44-SAME: name: "strides"
        CHECK-LLVMLITE-LE44-SAME: offset: 384
        CHECK-LLVMLITE-LE44-SAME: size: 64
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: [[DBG146:.+]] = !{ [[DBG129]], [[DBG132]], [[DBG134]], [[DBG136]], [[DBG139]], [[DBG142]], [[DBG145]] }
        CHECK-LLVMLITE-LE44: [[DBG147:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-LE44-SAME: elements: [[DBG146]]
        CHECK-LLVMLITE-LE44-SAME: identifier: "{i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}"
        CHECK-LLVMLITE-LE44-SAME: name: "array(float32, 1d, C) ({i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]})"
        CHECK-LLVMLITE-LE44-SAME: size: 448
        CHECK-LLVMLITE-LE44-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-LE44-SAME: )
        CHECK-LLVMLITE-LE44: !DILocalVariable(
        CHECK-LLVMLITE-LE44-SAME: name: "a"
        CHECK-LLVMLITE-LE44-SAME: type: [[DBG147]]
        CHECK-LLVMLITE-LE44-SAME: )

        CHECK-LLVMLITE-GE45: [[DBG123:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG98:![0-9]+]]
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG124:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG123]]
        CHECK-LLVMLITE-GE45-SAME: name: "meminfo"
        CHECK-LLVMLITE-GE45-SAME: offset: 0
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG125:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG98]]
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG126:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG125]]
        CHECK-LLVMLITE-GE45-SAME: name: "parent"
        CHECK-LLVMLITE-GE45-SAME: offset: 64
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG127:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG105:![0-9]+]]
        CHECK-LLVMLITE-GE45-SAME: name: "nitems"
        CHECK-LLVMLITE-GE45-SAME: offset: 128
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG128:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG105]]
        CHECK-LLVMLITE-GE45-SAME: name: "itemsize"
        CHECK-LLVMLITE-GE45-SAME: offset: 192
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG129:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG108:![0-9]+]]
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_pointer_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG130:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG129]]
        CHECK-LLVMLITE-GE45-SAME: name: "data"
        CHECK-LLVMLITE-GE45-SAME: offset: 256
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG131:.+]] = !DICompositeType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG111:![0-9]+]]
        CHECK-LLVMLITE-GE45-SAME: elements: [[DBG113:![0-9]+]]
        CHECK-LLVMLITE-GE45-SAME: identifier: "[1 x i64]"
        CHECK-LLVMLITE-GE45-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_array_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG132:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG131]]
        CHECK-LLVMLITE-GE45-SAME: name: "shape"
        CHECK-LLVMLITE-GE45-SAME: offset: 320
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG133:.+]] = !DICompositeType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG111]]
        CHECK-LLVMLITE-GE45-SAME: elements: [[DBG113]]
        CHECK-LLVMLITE-GE45-SAME: identifier: "[1 x i64]"
        CHECK-LLVMLITE-GE45-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_array_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG134:.+]] = !DIDerivedType(
        CHECK-LLVMLITE-GE45-SAME: baseType: [[DBG133]]
        CHECK-LLVMLITE-GE45-SAME: name: "strides"
        CHECK-LLVMLITE-GE45-SAME: offset: 384
        CHECK-LLVMLITE-GE45-SAME: size: 64
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_member
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG135:.+]] = !{ [[DBG124]], [[DBG126]], [[DBG127]], [[DBG128]], [[DBG130]], [[DBG132]], [[DBG134]] }
        CHECK-LLVMLITE-GE45: [[DBG136:.+]] = distinct !DICompositeType(
        CHECK-LLVMLITE-GE45-SAME: elements: [[DBG135]]
        CHECK-LLVMLITE-GE45-SAME: identifier: "{i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}"
        CHECK-LLVMLITE-GE45-SAME: name: "array(float32, 1d, C) ({i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]})"
        CHECK-LLVMLITE-GE45-SAME: size: 448
        CHECK-LLVMLITE-GE45-SAME: tag: DW_TAG_structure_type
        CHECK-LLVMLITE-GE45-SAME: )
        CHECK-LLVMLITE-GE45: [[DBG27:![0-9]+]] = !DILocalVariable(
        CHECK-LLVMLITE-GE45-SAME: name: "a"
        CHECK-LLVMLITE-GE45-SAME: type: [[DBG136]]
        CHECK-LLVMLITE-GE45-SAME: )
        """,
    ),
)


@skip_on_cudasim("Simulator does not produce debug dumps")
class TestCudaDebugInfoTypes(CUDATestCase):
    def test_ditypes(self):
        llvmlite_minor_version = int(llvmlite.__version__.split(".")[1])
        prefixes = (
            "CHECK",
            "CHECK-LLVMLITE-LE44"
            if llvmlite_minor_version <= 44
            else "CHECK-LLVMLITE-GE45",
        )

        def sanitize_name(name: str) -> str:
            return "".join(filter(lambda c: c.isalnum(), name))

        for numba_type, checks in scalar_types + array_types:
            with self.subTest(
                f"Test DITypes for {sanitize_name(numba_type.name)}"
            ):

                @cuda.jit((numba_type,), debug=True, opt=False)
                def foo(a):
                    pass

                ir = foo.inspect_llvm()[foo.signatures[0]]
                self.assertFileCheckMatches(ir, checks, check_prefixes=prefixes)
