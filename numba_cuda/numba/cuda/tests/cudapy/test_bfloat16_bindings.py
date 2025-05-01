import numba.cuda as cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np

from numba import int16, int32, int64, uint16, uint32, uint64, float32, float64
from numba.types import float16

from numba.cuda.cuda_bf16 import (
    nv_bfloat16,
    htrunc,
    hceil,
    hfloor,
    hrint,
    hsqrt,
    hrsqrt,
    hrcp,
    hlog,
    hlog2,
    hlog10,
    hcos,
    hsin,
    hexp,
    hexp2,
    hexp10,
)

from numba.cuda.cudadrv.runtime import get_version

cuda_version = get_version()

dtypes = [int16, int32, int64, uint16, uint32, uint64, float32]


@unittest.skipIf(
    (cuda.get_current_device().compute_capability < (8, 0)),
    "bfloat16 requires compute capability 8.0+",
)
class Bfloat16Test(CUDATestCase):
    def test_ctor(self):
        @cuda.jit
        def simple_kernel():
            a = nv_bfloat16(float64(1.0))  # noqa: F841
            b = nv_bfloat16(float32(2.0))  # noqa: F841
            c = nv_bfloat16(int16(3))  # noqa: F841
            d = nv_bfloat16(int32(4))  # noqa: F841
            e = nv_bfloat16(int64(5))  # noqa: F841
            f = nv_bfloat16(uint16(6))  # noqa: F841
            g = nv_bfloat16(uint32(7))  # noqa: F841
            h = nv_bfloat16(uint64(8))  # noqa: F841

        simple_kernel[1, 1]()

        if cuda_version >= (12, 0):

            @cuda.jit
            def simple_kernel_fp16():
                i = nv_bfloat16(float16(9))  # noqa: F841

            simple_kernel_fp16[1, 1]()

    def test_casts(self):
        @cuda.jit
        def simple_kernel(b, c, d, e, f, g, h):
            a = nv_bfloat16(3.14)

            b[0] = float32(a)
            c[0] = int16(a)
            d[0] = int32(a)
            e[0] = int64(a)
            f[0] = uint16(a)
            g[0] = uint32(a)
            h[0] = uint64(a)

        b = np.zeros(1, dtype=np.float32)
        c = np.zeros(1, dtype=np.int16)
        d = np.zeros(1, dtype=np.int32)
        e = np.zeros(1, dtype=np.int64)
        f = np.zeros(1, dtype=np.uint16)
        g = np.zeros(1, dtype=np.uint32)
        h = np.zeros(1, dtype=np.uint64)

        simple_kernel[1, 1](b, c, d, e, f, g, h)

        np.testing.assert_allclose(b[0], 3.14, atol=1e-2)
        assert c[0] == 3
        assert d[0] == 3
        assert e[0] == 3
        assert f[0] == 3
        assert g[0] == 3
        assert h[0] == 3

    def test_ctor_cast_loop(self):
        for dtype in dtypes:
            with self.subTest(dtype=dtype):

                @cuda.jit
                def simple_kernel(a):
                    a[0] = dtype(nv_bfloat16(dtype(3.14)))

                a = np.zeros(1, dtype=str(dtype))
                simple_kernel[1, 1](a)

                if np.dtype(str(dtype)).kind == "f":
                    np.testing.assert_allclose(a[0], 3.14, atol=1e-2)
                else:
                    assert a[0] == 3

    def test_arithmetic(self):
        @cuda.jit
        def simple_kernel(arith, logic):
            # Binary Arithmetic Operators
            a = nv_bfloat16(1.0)
            b = nv_bfloat16(2.0)

            arith[0] = float32(a + b)
            arith[1] = float32(a - b)
            arith[2] = float32(a * b)
            arith[3] = float32(a / b)

            # Arithmetic Assignment Operators
            a = nv_bfloat16(1.0)
            b = nv_bfloat16(2.0)

            a += b
            arith[4] = float32(a)
            a -= b
            arith[5] = float32(a)
            a *= b
            arith[6] = float32(a)
            a /= b
            arith[7] = float32(a)

            # Unary Arithmetic Operators
            a = nv_bfloat16(1.0)

            arith[8] = float32(+a)
            arith[9] = float32(-a)

            # Comparison Operators
            a = nv_bfloat16(1.0)
            b = nv_bfloat16(2.0)

            logic[0] = a == b
            logic[1] = a != b
            logic[2] = a > b
            logic[3] = a < b
            logic[4] = a >= b
            logic[5] = a <= b

        arith = np.zeros(10, dtype=np.float32)
        logic = np.zeros(6, dtype=np.bool_)

        simple_kernel[1, 1](arith, logic)

        a = 1.0
        b = 2.0
        np.testing.assert_allclose(
            arith,
            [
                a + b,
                a - b,
                a * b,
                a / b,
                a + b,
                a + b - b,
                (a + b - b) * b,
                (a + b - b) * b / b,
                +a,
                -a,
            ],
            atol=1e-2,
        )
        np.testing.assert_equal(
            logic, [a == b, a != b, a > b, a < b, a >= b, a <= b]
        )

    def test_math_func(self):
        @cuda.jit
        def simple_kernel(a):
            x = nv_bfloat16(3.14)

            a[0] = float32(htrunc(x))
            a[1] = float32(hceil(x))
            a[2] = float32(hfloor(x))
            a[3] = float32(hrint(x))
            a[4] = float32(hsqrt(x))
            a[5] = float32(hrsqrt(x))
            a[6] = float32(hrcp(x))
            a[7] = float32(hlog(x))
            a[8] = float32(hlog2(x))
            a[9] = float32(hlog10(x))
            a[10] = float32(hcos(x))
            a[11] = float32(hsin(x))
            a[12] = float32(hexp(x))
            a[13] = float32(hexp2(x))
            a[14] = float32(hexp10(x))

        a = np.zeros(15, dtype=np.float32)
        simple_kernel[1, 1](a)

        x = 3.14
        np.testing.assert_allclose(
            a[:12],
            [
                np.trunc(x),
                np.ceil(x),
                np.floor(x),
                np.rint(x),
                np.sqrt(x),
                1 / np.sqrt(x),
                1 / x,
                np.log(x),
                np.log2(x),
                np.log10(x),
                np.cos(x),
                np.sin(x),
            ],
            atol=1e-2,
        )

        np.testing.assert_allclose(
            a[12:], [np.exp(x), np.exp2(x), np.power(10, x)], atol=1e2
        )

    def test_check_bfloat16_type(self):
        @cuda.jit
        def kernel(arr):
            x = nv_bfloat16(3.14)
            if isinstance(x, nv_bfloat16):
                arr[0] = float32(x)
            else:
                arr[0] = float32(0.0)

        arr = np.zeros(1, np.float32)
        kernel[1, 1](arr)

        np.testing.assert_allclose(arr, [3.14], atol=1e-2)

    def test_use_within_device_func(self):
        @cuda.jit(device=True)
        def add_bf16(a, b):
            return a + b

        @cuda.jit
        def kernel(arr):
            a = nv_bfloat16(3.14)
            b = nv_bfloat16(5)
            arr[0] = float32(hfloor(add_bf16(a, b)))

        arr = np.zeros(1, np.float32)
        kernel[1, 1](arr)

        np.testing.assert_allclose(arr, [8], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
