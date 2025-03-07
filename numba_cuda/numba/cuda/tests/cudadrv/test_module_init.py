import numpy as np

from numba import cuda
from numba.cuda.cudadrv.linkable_code import CUSource
from numba.cuda.testing import CUDATestCase

from cuda.bindings.driver import (
    cuModuleGetGlobal,
    cuMemcpyHtoD,
    cuLibraryGetModule
)


class TestModuleInitCallback(CUDATestCase):

    def setUp(self):
        super().setUp()

        module = """
__device__ int num = 0;
extern "C"
__device__ int get_num(int &retval) {
    retval = num;
    return 0;
}
"""

        def set_fourty_two(obj):
            # Initialize 42 to global variable `num`
            culib = obj._handle
            res, mod = cuLibraryGetModule(culib)
            self.assertEqual(res, 0)

            res, dptr, size = cuModuleGetGlobal(mod, "num".encode())
            self.assertEqual(res, 0)
            self.assertEqual(size, 4)

            arr = np.array([42], np.int32)
            cuMemcpyHtoD(dptr, arr.ctypes.data, size)

        self.lib = CUSource(module, init_callback=set_fourty_two)

    def test_decldevice_arg(self):
        get_num = cuda.declare_device("get_num", "int32()", link=[self.lib])

        @cuda.jit
        def kernel(arr):
            arr[0] = get_num()

        arr = np.zeros(1, np.int32)
        kernel[1, 1](arr)
        self.assertEqual(arr[0], 42)

    def test_jitarg(self):
        get_num = cuda.declare_device("get_num", "int32()")

        @cuda.jit(link=[self.lib])
        def kernel(arr):
            arr[0] = get_num()

        arr = np.zeros(1, np.int32)
        kernel[1, 1](arr)
        self.assertEqual(arr[0], 42)
