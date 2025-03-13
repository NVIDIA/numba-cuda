import gc

import numpy as np

from numba import cuda
from numba.cuda.cudadrv.linkable_code import CUSource
from numba.cuda.testing import CUDATestCase

from cuda.bindings.driver import cuModuleGetGlobal, cuMemcpyHtoD


class TestModuleCallbacksBasic(CUDATestCase):

    def test_basic(self):
        counter = [0]

        def setup(mod, counter=counter):
            counter[0] += 1

        def teardown(mod, counter=counter):
            counter[0] -= 1

        lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

        @cuda.jit(link=[lib])
        def kernel():
            pass

        self.assertEqual(counter, [0])
        kernel[1, 1]()
        self.assertEqual(counter, [1])
        kernel[1, 1]() # cached
        self.assertEqual(counter, [1])
        breakpoint()
        del kernel
        gc.collect()
        # When does the cache gets cleared?
        self.assertEqual(counter, [0]) # FAILS


class TestModuleCallbacks(CUDATestCase):

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

        self.counter = 0

        def set_forty_two(mod):
            self.counter += 1
            # Initialize 42 to global variable `num`
            res, dptr, size = cuModuleGetGlobal(
                mod.handle.value, "num".encode()
            )

            arr = np.array([42], np.int32)
            cuMemcpyHtoD(dptr, arr.ctypes.data, size)

        def teardown(mod):
            self.counter -= 1

        self.lib = CUSource(
            module, setup_callback=set_forty_two, teardown_callback=teardown)

    def tearDown(self):
        super().tearDown()
        del self.lib

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
