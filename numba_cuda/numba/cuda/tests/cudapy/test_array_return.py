import numpy as np
import pytest

from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba import cuda
from numba.core.errors import TypingError


class TestCudaArrayReturn(CUDATestCase):
    def _test_array_return(self, test_function):
        array_return = cuda.jit(test_function)

        @cuda.jit
        def kernel(x, y):
            y = array_return(x, y)
            y[2] = 1

        a = np.zeros(5)
        b = np.ones(5)

        kernel[1, 1](a, b)

        assert a[0] == 0
        assert a[2] == 1
        assert all(b == 1)

    def test_array_return(self):
        def array_return(a, b):
            return a

        self._test_array_return(array_return)

    def test_array_return_conditional(self):
        def array_return_conditional(a, b):
            if b[0] > a[0]:
                r = a
            else:
                r = b
            return r

        self._test_array_return(array_return_conditional)

    def test_array_return_deeper_alias(self):
        def array_return_deeper_alias(a, b):
            if len(a) >= len(b):
                tmp = a
            else:
                tmp = b
            result = tmp
            return result

        self._test_array_return(array_return_deeper_alias)

    def test_array_return_for_break(self):
        def array_return_for_break(x, y):
            found = None
            for arr in (x, y):
                if sum(arr) == 0:
                    found = arr
                    break
            return found

        self._test_array_return(array_return_for_break)

    def test_array_return_while_continue(self):
        def array_return_while_continue(x, y):
            arrays = (x, y)
            i = 0
            while i < len(arrays):
                if len(arrays[i]) == 0:
                    i += 1
                    continue
                if arrays[i][0] == 0:
                    selected = arrays[i]
                    return selected
                i += 1
            return x

        self._test_array_return(array_return_while_continue)

    def test_array_return_nested_if(self):
        def array_return_nested_if(x, y):
            if len(x) > 2:
                if x[0] < 0:
                    temp = y
                else:
                    temp = x
                ans = temp
            else:
                ans = x
            return ans

        self._test_array_return(array_return_nested_if)

    def test_array_return_alias_loop(self):
        def array_return_alias_loop(a, b):
            for arr in (a, b):
                alias = arr
                if len(alias) > 1:
                    return alias
            return b

        self._test_array_return(array_return_alias_loop)

    def test_array_return_from_tuple(self):
        def array_return_from_tuple(a, b):
            t = (a, b)
            return t[0]

        self._test_array_return(array_return_from_tuple)

    def _test_array_slice(self, test_function):
        array_slice = cuda.jit(test_function)

        @cuda.jit
        def kernel(x):
            y = array_slice(x, 2, 3)
            y[0] = 1

        a = np.zeros(5)

        kernel[1, 1](a)

        assert a[0] == 0
        assert a[2] == 1

    def test_array_slice(self):
        def array_slice(a, start, end):
            return a[start:end]

        self._test_array_slice(array_slice)

    def test_array_slice_conditional(self):
        def array_slice_conditional(a, start, end):
            if start > 0:
                y = a[start:end]
            else:
                y = a[end:start]
            return y

        self._test_array_slice(array_slice_conditional)

    def test_array_slice_2d(self):
        @cuda.jit
        def array_slice_2d(a, x_id, x_size, y_id, y_size):
            return a[
                x_id * x_size : (x_id + 1) * x_size : 1,
                y_id * y_size : (y_id + 1) * y_size : 1,
            ]

        @cuda.jit
        def kernel(x):
            y = array_slice_2d(x, 1, 2, 2, 2)
            y[0, 0] = 1

        a = np.zeros((4, 6))

        kernel[1, 1](a)

        assert a[0, 0] == 0
        assert a[2, 4] == 1

    def test_view_return(self):
        @cuda.jit
        def array_return(a):
            return a.view(np.uint16)

        @cuda.jit
        def kernel(x):
            y = array_return(x)
            y[0] = 1
            y[1] = 2

        a = np.zeros(2, np.uint32)
        a[1] = 3

        kernel[1, 1](a)

        assert a[0] == (2**16) * 2 + 1
        assert a[1] == 3

    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_local_illegal(self):
        @cuda.jit
        def array_local(shape, dtype):
            return cuda.local.array(shape, dtype=dtype)

        @cuda.jit
        def kernel():
            x = array_local((2, 3), np.int32)
            x[0, 0] = 1

        with self.assertRaises(TypingError):
            kernel[1, 1]()

    @pytest.mark.xfail(reason="Returning const arrays is not yet supported")
    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_const(self):
        const_data = np.asarray([1, 2, 3])

        @cuda.jit
        def array_const():
            return cuda.const.array_like(const_data)

        r = np.zeros_like(const_data)

        @cuda.jit
        def kernel(r):
            x = array_const()
            i = cuda.grid(1)

            if i < len(r):
                r[i] = x[i]

        kernel[1, 1](r)

        np.testing.assert_equal(r, const_data)

    @pytest.mark.xfail(reason="Returning local arrays is not yet supported")
    @skip_on_cudasim("type inference is unsupported in the simulator")
    def test_array_local(self):
        @cuda.jit
        def array_local_fp32(size):
            return cuda.local.array(size, dtype=np.float32)

        @cuda.jit
        def kernel(r):
            x = array_local_fp32(2)
            x[0], x[1] = 1.0, 2.0

            r[0] = x[0] + x[1]

        r = np.zeros(1, dtype=np.float32)

        kernel[1, 1](r)

        np.testing.assert_equal(r, [3.0])


if __name__ == "__main__":
    unittest.main()
