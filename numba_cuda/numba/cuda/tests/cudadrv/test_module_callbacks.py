import gc

import numpy as np

from numba import cuda
from numba.cuda.cudadrv.linkable_code import CUSource
from numba.cuda.testing import CUDATestCase

from cuda.bindings.driver import cuModuleGetGlobal, cuMemcpyHtoD


class TestModuleCallbacksBasic(CUDATestCase):

    def test_basic(self):
        counter = 0

        def setup(mod, stream):
            nonlocal counter
            counter += 1

        def teardown(mod, stream):
            nonlocal counter
            counter -= 1

        lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

        @cuda.jit(link=[lib])
        def kernel():
            pass

        self.assertEqual(counter, 0)
        kernel[1, 1]()
        self.assertEqual(counter, 1)
        kernel[1, 1]() # cached
        self.assertEqual(counter, 1)
        breakpoint()
        del kernel
        gc.collect()
        cuda.current_context().deallocations.clear()
        self.assertEqual(counter, 0)
        # We don't have a way to explicitly evict kernel and its modules at
        # the moment.

    def test_different_argtypes(self):
        counter = 0

        def setup(mod, stream):
            nonlocal counter
            counter += 1

        def teardown(mod, stream):
            nonlocal counter
            counter -= 1

        lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

        @cuda.jit(link=[lib])
        def kernel(arg):
            pass

        self.assertEqual(counter, 0)
        kernel[1, 1](42)    # (int64)->() : module 1
        self.assertEqual(counter, 1)
        kernel[1, 1](100)   # (int64)->() : module 1, cached
        self.assertEqual(counter, 1)
        kernel[1, 1](3.14)  # (float64)->() : module 2
        self.assertEqual(counter, 2)

        # del kernel
        # gc.collect()
        # cuda.current_context().deallocations.clear()
        # self.assertEqual(counter, 0) # We don't have a way to explicitly
        # evict kernel and its modules at the moment.

    def test_two_kernels(self):
        counter = 0

        def setup(mod, stream):
            nonlocal counter
            counter += 1

        def teardown(mod, stream):
            nonlocal counter
            counter -= 1

        lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

        @cuda.jit(link=[lib])
        def kernel():
            pass

        @cuda.jit(link=[lib])
        def kernel2():
            pass

        kernel[1, 1]()
        self.assertEqual(counter, 1)
        kernel2[1, 1]()
        self.assertEqual(counter, 2)

        # del kernel
        # gc.collect()
        # cuda.current_context().deallocations.clear()
        # self.assertEqual(counter, 0) # We don't have a way to explicitly
        # evict kernel and its modules at the moment.


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

        def set_forty_two(mod, stream):
            self.counter += 1
            # Initialize 42 to global variable `num`
            res, dptr, size = cuModuleGetGlobal(
                mod.handle.value, "num".encode()
            )

            arr = np.array([42], np.int32)
            cuMemcpyHtoD(dptr, arr.ctypes.data, size)

        def teardown(mod, stream):
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
