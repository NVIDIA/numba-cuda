import itertools
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase
import unittest


# Set to true if you want to see dots printed for each subtest.
NOISY = True


# N.B. We name the test class TestArrayAddressAlignment to avoid name conflict
#      with the test_alignment.TestArrayAlignment class.


class TestArrayAddressAlignment(CUDATestCase):
    """
    Test cuda.local.array and cuda.shared.array support for an alignment
    keyword argument.
    """

    def test_array_alignment(self):
        shapes = (1, 8, 50)
        dtypes = (np.uint8, np.uint32, np.uint64)
        alignments = (None, 16, 64, 256)
        array_types = [(0, 'local'), (1, 'shared')]

        items = itertools.product(array_types, shapes, dtypes, alignments)

        for (which, array_type), shape, dtype, alignment in items:
            with self.subTest(array_type=array_type, shape=shape,
                              dtype=dtype, alignment=alignment):
                @cuda.jit
                def f(loc, shrd, which):
                    i = cuda.grid(1)
                    if which == 0:
                        local_array = cuda.local.array(
                            shape=shape,
                            dtype=dtype,
                            alignment=alignment,
                        )
                        if i == 0:
                            loc[0] = local_array.ctypes.data
                    else:
                        shared_array = cuda.shared.array(
                            shape=shape,
                            dtype=dtype,
                            alignment=alignment,
                        )
                        if i == 0:
                            shrd[0] = shared_array.ctypes.data

                loc = np.zeros(1, dtype=np.uint64)
                shrd = np.zeros(1, dtype=np.uint64)
                f[1, 1](loc, shrd, which)

                if alignment is not None:
                    address = loc[0] if which == 0 else shrd[0]
                    alignment_mod = int(address % alignment)
                    self.assertEqual(alignment_mod, 0)

                if NOISY:
                    print('.', end='', flush=True)

    def test_invalid_aligments(self):
        shapes = (1, 50)  # Just test small and large
        dtypes = (np.uint8, np.uint64)  # Just test smallest and largest
        # Keep a good selection of invalid alignments to test error cases
        alignments = (-1, 0, 3, 17, 33)  # Negative, zero, and non-powers of 2
        array_types = [(0, 'local'), (1, 'shared')]

        # This reduces from 576 to 40 test cases (2×2×2×5)
        items = itertools.product(array_types, shapes, dtypes, alignments)

        for (which, array_type), shape, dtype, alignment in items:
            with self.subTest(array_type=array_type, shape=shape,
                              dtype=dtype, alignment=alignment):
                @cuda.jit
                def f(local_array, shared_array, which):
                    i = cuda.grid(1)
                    if which == 0:
                        local_array = cuda.local.array(
                            shape=shape,
                            dtype=dtype,
                            alignment=alignment,
                        )
                        if i == 0:
                            local_array[0] = local_array.ctypes.data
                    else:
                        shared_array = cuda.shared.array(
                            shape=shape,
                            dtype=dtype,
                            alignment=alignment,
                        )
                        if i == 0:
                            shared_array[0] = shared_array.ctypes.data

                loc = np.zeros(1, dtype=np.uint64)
                shrd = np.zeros(1, dtype=np.uint64)

                with self.assertRaises(ValueError) as raises:
                    f[1, 1](loc, shrd, which)
                exc = str(raises.exception)
                self.assertIn("Alignment must be", exc)

                if NOISY:
                    print('.', end='', flush=True)

    def test_array_like(self):
        # XXX-140: TODO; need to flush out the array_like stuff more.
        pass


if __name__ == '__main__':
    unittest.main()
