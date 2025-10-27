# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from numba.cuda import types, config
from numba.core import ir
from numba.cuda import compiler
from numba.cuda.core.annotations import type_annotations
from numba.cuda.core.ir_utils import (
    copy_propagate,
    apply_copy_propagate,
    get_name_var_table,
)
from numba.cuda.core.typed_passes import type_inference_stage
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest


def _test_will_propagate(b, z, w):
    x = 3
    x1 = x
    if b > 0:
        y = z + w  # noqa: F821
    else:
        y = 0  # noqa: F841
    a = 2 * x1
    return a < b


def _test_wont_propagate(b, z, w):
    x = 3
    if b > 0:
        y = z + w  # noqa: F841
        x = 1
    else:
        y = 0  # noqa: F841
    a = 2 * x
    return a < b


def _in_list_var(list_var, var):
    for i in list_var:
        if i.name == var:
            return True
    return False


def _find_assign(func_ir, var):
    for label, block in func_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name != var:
                all_var = inst.list_vars()
                if _in_list_var(all_var, var):
                    return True

    return False


@skip_on_cudasim("cudasim doesn't support run_frontend")
class TestCopyPropagate(CUDATestCase):
    def test1(self):
        from numba.cuda.descriptor import cuda_target

        typingctx = cuda_target.typing_context
        targetctx = cuda_target.target_context
        test_ir = compiler.run_frontend(_test_will_propagate)
        typingctx.refresh()
        targetctx.refresh()
        args = (types.int64, types.int64, types.int64)
        typemap, return_type, calltypes, _ = type_inference_stage(
            typingctx, targetctx, test_ir, args, None
        )
        _ = type_annotations.TypeAnnotation(
            func_ir=test_ir,
            typemap=typemap,
            calltypes=calltypes,
            lifted=(),
            lifted_from=None,
            args=args,
            return_type=return_type,
            html_output=config.HTML,
        )
        in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
        _ = apply_copy_propagate(
            test_ir.blocks,
            in_cps,
            get_name_var_table(test_ir.blocks),
            typemap,
            calltypes,
        )

        self.assertFalse(_find_assign(test_ir, "x1"))

    def test2(self):
        from numba.cuda.descriptor import cuda_target

        typingctx = cuda_target.typing_context
        targetctx = cuda_target.target_context
        test_ir = compiler.run_frontend(_test_wont_propagate)
        typingctx.refresh()
        targetctx.refresh()
        args = (types.int64, types.int64, types.int64)
        typemap, return_type, calltypes, _ = type_inference_stage(
            typingctx, targetctx, test_ir, args, None
        )
        _ = type_annotations.TypeAnnotation(
            func_ir=test_ir,
            typemap=typemap,
            calltypes=calltypes,
            lifted=(),
            lifted_from=None,
            args=args,
            return_type=return_type,
            html_output=config.HTML,
        )
        in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
        _ = apply_copy_propagate(
            test_ir.blocks,
            in_cps,
            get_name_var_table(test_ir.blocks),
            typemap,
            calltypes,
        )

        self.assertTrue(_find_assign(test_ir, "x"))


if __name__ == "__main__":
    unittest.main()
