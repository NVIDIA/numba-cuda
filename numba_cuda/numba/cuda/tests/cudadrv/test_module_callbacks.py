import unittest

import numpy as np

from numba import cuda, config
from numba.cuda.cudadrv.linkable_code import CUSource
from numba.cuda.testing import CUDATestCase

from cuda.bindings.driver import cuModuleGetGlobal, cuMemcpyHtoD


def wipe_all_modules_in_context():
    ctx = cuda.current_context()
    ctx.modules.clear()


@unittest.skipIf(
    config.CUDA_USE_NVIDIA_BINDING,
    "NV binding support superceded by cuda.bindings."
)
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

        wipe_all_modules_in_context()
        self.assertEqual(counter, 0)

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

        wipe_all_modules_in_context()
        self.assertEqual(counter, 0)

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

        wipe_all_modules_in_context()
        self.assertEqual(counter, 0)


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
