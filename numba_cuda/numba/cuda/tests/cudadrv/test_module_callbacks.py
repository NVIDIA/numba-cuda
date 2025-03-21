import unittest

import numpy as np

from numba import cuda, config
from numba.cuda.cudadrv.linkable_code import CUSource
from numba.cuda.testing import CUDATestCase, ContextResettingTestCase

from cuda.bindings.driver import cuModuleGetGlobal, cuMemcpyHtoD

if config.CUDA_USE_NVIDIA_BINDING:
    from cuda.cuda import CUmodule as cu_module_type
else:
    from numba.cuda.cudadrv.drvapi import cu_module as cu_module_type


def wipe_all_modules_in_context():
    ctx = cuda.current_context()
    ctx.reset()


def get_hashable_handle_value(handle):
    if not config.CUDA_USE_NVIDIA_BINDING:
        handle = handle.value
    return handle


class TestModuleCallbacksBasic(ContextResettingTestCase):

    def test_basic(self):
        counter = 0

        def setup(handle):
            self.assertTrue(isinstance(handle, cu_module_type))
            nonlocal counter
            counter += 1

        def teardown(handle):
            self.assertTrue(isinstance(handle, cu_module_type))
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
        del kernel
        self.assertEqual(counter, 0)

    def test_different_argtypes(self):
        counter = 0
        setup_seen = set()
        teardown_seen = set()

        def setup(handle):
            nonlocal counter, setup_seen
            counter += 1
            setup_seen.add(get_hashable_handle_value(handle))

        def teardown(handle):
            nonlocal counter
            counter -= 1
            teardown_seen.add(get_hashable_handle_value(handle))

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
        del kernel
        self.assertEqual(counter, 0)

        self.assertEqual(len(setup_seen), 2)
        self.assertEqual(len(teardown_seen), 2)

    def test_two_kernels(self):
        counter = 0
        setup_seen = set()
        teardown_seen = set()

        def setup(handle):
            nonlocal counter, setup_seen
            counter += 1
            setup_seen.add(get_hashable_handle_value(handle))

        def teardown(handle):
            nonlocal counter, teardown_seen
            counter -= 1
            teardown_seen.add(get_hashable_handle_value(handle))

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
        del kernel
        self.assertEqual(counter, 0)

        self.assertEqual(len(setup_seen), 2)
        self.assertEqual(len(teardown_seen), 2)


class TestModuleCallbacksAPICompleteness(CUDATestCase):

    def test_api(self):
        def setup(handle):
            pass

        def teardown(handle):
            pass

        api_combo = [
            (setup, teardown),
            (setup, None),
            (None, teardown),
            (None, None)
        ]

        for setup, teardown in api_combo:
            with self.subTest(setup=setup, teardown=teardown):
                lib = CUSource(
                    "", setup_callback=setup, teardown_callback=teardown)

                @cuda.jit(link=[lib])
                def kernel():
                    pass

                kernel[1, 1]()


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

        def set_forty_two(handle):
            # Initialize 42 to global variable `num`
            res, dptr, size = cuModuleGetGlobal(
                get_hashable_handle_value(handle), "num".encode()
            )

            arr = np.array([42], np.int32)
            cuMemcpyHtoD(dptr, arr.ctypes.data, size)

        self.lib = CUSource(
            module, setup_callback=set_forty_two, teardown_callback=None)

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


if __name__ == '__main__':
    unittest.main()
