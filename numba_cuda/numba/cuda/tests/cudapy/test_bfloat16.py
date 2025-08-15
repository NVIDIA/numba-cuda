from numba import cuda, float32
from numba.cuda.bf16 import (
    bfloat16,
    habs,
    hadd,
    hsub,
    hmul,
    hadd_rn,
    hsub_rn,
    hmul_rn,
    hdiv,
    hadd_sat,
    hsub_sat,
    hmul_sat,
    hfma,
    hfma_sat,
    hneg,
    hfma_relu,
)
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

    def test_arithmetic_intrinsics_basic(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(1.25)
            b = bfloat16(-2.5)

            out[0] = float32(habs(b))
            out[1] = float32(hadd(a, b))
            out[2] = float32(hsub(a, b))
            out[3] = float32(hmul(a, b))
            out[4] = float32(hdiv(b, a))
            out[5] = float32(hneg(a))
            out[6] = float32(hfma(a, b, b))

            out[7] = float32(hadd_rn(a, b))
            out[8] = float32(hsub_rn(a, b))
            out[9] = float32(hmul_rn(a, b))

        out = cuda.device_array((10,), dtype="float32")
        kernel[1, 1](out)

        a = 1.25
        b = -2.5
        expected = [
            abs(b),
            a + b,
            a - b,
            a * b,
            b / a,
            -a,
            a * b + b,
            a + b,
            a - b,
            a * b,
        ]
        for i, exp in enumerate(expected):
            self.assertAlmostEqual(out[i], exp, delta=1e-2)

    def test_arithmetic_intrinsics_saturating(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(1.5)
            b = bfloat16(0.75)

            out[0] = float32(hadd_sat(a, b))  # 2.25 -> 1.0
            out[1] = float32(hsub_sat(b, a))  # -0.75 -> 0.0
            out[2] = float32(hmul_sat(a, b))  # 1.125 -> 1.0
            out[3] = float32(hfma_sat(a, b, a))  # 1.125 + 1.5 -> 1.0

        out = cuda.device_array((4,), dtype="float32")
        kernel[1, 1](out)

        self.assertAlmostEqual(out[0], 1.0, delta=1e-3)
        self.assertAlmostEqual(out[1], 0.0, delta=1e-3)
        self.assertAlmostEqual(out[2], 1.0, delta=1e-3)
        self.assertAlmostEqual(out[3], 1.0, delta=1e-3)

        # Also check they are clamped within [0, 1]
        for i in range(4):
            self.assertGreaterEqual(out[i], 0.0)
            self.assertLessEqual(out[i], 1.0)

    def test_fma_relu_intrinsic(self):
        self.skip_unsupported()

        @cuda.jit
        def kernel(out):
            a = bfloat16(-1.5)
            b = bfloat16(2.0)
            c = bfloat16(0.0)

            out[0] = float32(hfma_relu(a, b, c))  # -3.0 -> relu -> 0.0

        out = cuda.device_array((1,), dtype="float32")
        kernel[1, 1](out)

        self.assertAlmostEqual(out[0], 0.0, delta=1e-3)
