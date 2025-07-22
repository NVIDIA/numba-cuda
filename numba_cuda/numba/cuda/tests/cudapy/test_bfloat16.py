from numba import cuda, float32
from numba.cuda.bf16 import bfloat16
from numba.cuda.testing import CUDATestCase

import math


class TestBfloat16HighLevelBindings(CUDATestCase):
    def skip_unsupported(self):
        if not cuda.is_bfloat16_supported():
            self.skipTest(
                "bfloat16 requires compute capability 8.0+ and CUDA version>= 12.0"
            )

    def test_use_type_in_kernel(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel():
            bfloat16(3.14)

        kernel[1, 1]()

    def test_math_bindings(self):
        self.skip_unsupported()

        exp_functions = [math.exp]
        try:
            from math import exp2

            exp_functions += [exp2]
        except ImportError:
            pass

        functions = [
            math.trunc,
            math.ceil,
            math.floor,
            math.sqrt,
            math.log,
            math.log10,
            math.cos,
            math.sin,
            math.tanh,
        ] + exp_functions

        for f in functions:
            with self.subTest(func=f):

                @cuda.jit
                def kernel(arr):
                    x = bfloat16(3.14)
                    y = f(x)
                    arr[0] = float32(y)

                arr = cuda.device_array((1,), dtype="float32")
                kernel[1, 1](arr)

                if f in exp_functions:
                    self.assertAlmostEqual(arr[0], f(3.14), delta=1e-1)
                else:
                    self.assertAlmostEqual(arr[0], f(3.14), delta=1e-2)
