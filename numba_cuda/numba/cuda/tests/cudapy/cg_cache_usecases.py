from numba import cuda
from numba.cuda.testing import CUDATestCase
import sys

from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase


# Usecase with cooperative groups


@cuda.jit(cache=True)
def cg_usecase_kernel(r, x):
    grid = cuda.cg.this_grid()
    grid.sync()


cg_usecase = CUDAUseCase(cg_usecase_kernel)


class _TestModule(CUDATestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        mod.cg_usecase(0)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
