# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import numpy as np
from numba import cuda
from numba.cuda import types
from numba.cuda.testing import (
    unittest,
    CUDATestCase,
    skip_on_cudasim,
)


@skip_on_cudasim("Cudasim does not support inline and forceinline")
class TestCudaInline(CUDATestCase):
    def _test_call_inline(self, inline, inline_expected):
        """Test @cuda.jit(inline=...)"""
        a = np.ones(2, dtype=np.int32)

        sig = (types.int32[::1],)

        @cuda.jit(inline=inline)
        def set_zero(a):
            a[0] = 0

        @cuda.jit(sig)
        def call_set_zero(a):
            set_zero(a)

        call_set_zero[1, 2](a)

        expected = np.arange(2, dtype=np.int32)
        self.assertTrue(np.all(a == expected))

        llvm_ir = call_set_zero.inspect_llvm(sig)
        pat = r"call [a-zA-Z0-9]* @"
        match = re.compile(pat).search(llvm_ir)

        if inline_expected:
            # check that call was inlined
            self.assertIsNone(match, msg=llvm_ir)
        else:
            # check that call was not inlined
            self.assertIsNotNone(match, msg=llvm_ir)

        # alwaysinline should not be in the IR when the inline kwarg is used
        self.assertNotIn("alwaysinline", llvm_ir)

    def test_call_inline_always(self):
        self._test_call_inline("always", True)

    def test_call_inline_never(self):
        self._test_call_inline("never", False)

    def test_call_inline_true(self):
        self._test_call_inline(True, True)

    def test_call_inline_false(self):
        self._test_call_inline(False, False)

    def test_call_inline_costmodel_false(self):
        def cost_model(expr, caller_info, callee_info):
            return False

        self._test_call_inline(cost_model, False)

    def test_call_inline_costmodel_true(self):
        def cost_model(expr, caller_info, callee_info):
            return True

        self._test_call_inline(cost_model, True)

    def _test_call_forceinline(self, forceinline):
        """Test @cuda.jit(forceinline=...)"""
        a = np.ones(2, dtype=np.int32)

        sig = (types.int32[::1],)

        @cuda.jit(forceinline=forceinline)
        def set_zero(a):
            a[0] = 0

        @cuda.jit(sig)
        def call_set_zero(a):
            set_zero(a)

        call_set_zero[1, 2](a)

        expected = np.arange(2, dtype=np.int32)
        self.assertTrue(np.all(a == expected))

        llvm_ir = call_set_zero.inspect_llvm(sig)
        pat = r"call [a-zA-Z0-9]* @"
        match = re.compile(pat).search(llvm_ir)

        # Check that call was not inlined at the Numba IR level - the call
        # should still be present in the IR
        self.assertIsNotNone(match)

        # Check the definition of set_zero - it is a definition where the
        # name does not include an underscore just before "set_zero", because
        # that would match the "call_set_zero" definition
        pat = r"define.*[^_]set_zero.*"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match)
        if forceinline:
            self.assertIn("alwaysinline", match.group())
        else:
            self.assertNotIn("alwaysinline", match.group())

        # The kernel, "call_set_zero", should never have "alwaysinline" set
        pat = r"define.*call_set_zero.*"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match)
        self.assertNotIn("alwaysinline", match.group())

    def test_call_forceinline_true(self):
        self._test_call_forceinline(True)

    def test_call_forceinline_false(self):
        self._test_call_forceinline(False)

    def test_compile_forceinline_ltoir_only(self):
        def set_zero(a):
            a[0] = 0

        args = (types.float32[::1],)
        msg = r"Can only designate forced inlining in LTO-IR"
        with self.assertRaisesRegex(ValueError, msg):
            cuda.compile(
                set_zero,
                args,
                device=True,
                forceinline=True,
            )

    def _compile_set_zero(self, forceinline):
        def set_zero(a):
            a[0] = 0

        args = (types.float32[::1],)
        ltoir, resty = cuda.compile(
            set_zero,
            args,
            device=True,
            output="ltoir",
            forceinline=forceinline,
        )

        # Sanity check
        self.assertEqual(resty, types.none)

        return ltoir

    def test_compile_forceinline(self):
        ltoir_noinline = self._compile_set_zero(False)
        ltoir_forceinline = self._compile_set_zero(True)

        # As LTO-IR is opaque, the best we can do is check that changing the
        # flag resulted in a change in the generated LTO-IR in some way.
        self.assertNotEqual(
            ltoir_noinline,
            ltoir_forceinline,
            "forceinline flag appeared to have no effect on LTO-IR",
        )


if __name__ == "__main__":
    unittest.main()
