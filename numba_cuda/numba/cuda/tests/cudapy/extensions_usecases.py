# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda import types
from numba.cuda.core import config


class MyStruct:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class StructModelType(types.Type):
    def __init__(self):
        super().__init__(name="TestStructModelType")


struct_model_type = StructModelType()


if not config.ENABLE_CUDASIM:
    from numba.cuda import int32
    from numba.cuda.extending import (
        core_models,
        typeof_impl,
        type_callable,
    )
    from numba.cuda.extending import (
        register_model,
        make_attribute_wrapper,
    )
    from numba.cuda.cudaimpl import lower
    from numba.cuda import cgutils

    @typeof_impl.register(MyStruct)
    def typeof_teststruct(val, c):
        return struct_model_type

    @register_model(StructModelType)
    class TestStructModel(core_models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [("x", int32), ("y", int32)]
            super().__init__(dmm, fe_type, members)

    make_attribute_wrapper(StructModelType, "x", "x")
    make_attribute_wrapper(StructModelType, "y", "y")

    @type_callable(MyStruct)
    def type_test_struct(context):
        def typer(x, y):
            if isinstance(x, types.Integer) and isinstance(y, types.Integer):
                return struct_model_type

        return typer

    @lower(MyStruct, types.Integer, types.Integer)
    def lower_test_type_ctor(context, builder, sig, args):
        obj = cgutils.create_struct_proxy(struct_model_type)(context, builder)
        obj.x = args[0]
        obj.y = args[1]
        return obj._getvalue()
