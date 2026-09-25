# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import unittest

import numpy as np

from numba import cuda, types
from numba.cuda.compiler import _compile_pyfunc_with_fixup
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.support import override_config


class TestCudaJitABI(CUDATestCase):
    """
    Tests the jit decorator with abi set to "C"
    """

    def test_abi_c(self):
        def normalize_llvm(s):
            return re.sub(
                r'(; ModuleID = ".*?\$)\d+(")',
                r"\g<1>X\g<2>",
                s,
                count=1,
            )

        def foo(a, b):
            return a + b

        sig = (types.int8, types.int8)

        x = cuda.jit(sig, device=True, abi="c")(foo).inspect_llvm()[(sig)]
        y = _compile_pyfunc_with_fixup(foo, sig=sig, abi="c", device=True)[
            0
        ].get_llvm_str()
        self.assertTrue(normalize_llvm(x) == normalize_llvm(y))
