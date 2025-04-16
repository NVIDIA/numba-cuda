import re
import numpy as np
from numba import cuda, types
from numba.cuda.testing import (
    unittest,
    CUDATestCase,
    skip_on_cudasim,
)


class TestCudaInline(CUDATestCase):
    @skip_on_cudasim("Cudasim does not support inline")
    def _test_call_inline(self, inline):
        """Test @cuda.jit(inline=...)"""
        a = np.ones(2, dtype=np.int32)

        sig = (types.int32[::1],)

        @cuda.jit(inline=inline)
        def set_zero(a):
            a[0] = 0

        @cuda.jit(sig)
        def call_set_zero(a):
            set_zero(a)

        call_set_zero[1, 2](a)

        expected = np.arange(2, dtype=np.int32)
        self.assertTrue(np.all(a == expected))

        llvm_ir = call_set_zero.inspect_llvm(sig)
        pat = r"call [a-zA-Z0-9]* @"
        match = re.compile(pat).search(llvm_ir)

        if inline == "always" or inline is True:
            # check that call was inlined
            self.assertIsNone(match, msg=llvm_ir)
        else:
            assert inline == "never" or inline is False

            # check that call was not inlined
            self.assertIsNotNone(match, msg=llvm_ir)

    def test_call_inline_always(self):
        self._test_call_inline("always")

    def test_call_inline_never(self):
        self._test_call_inline("never")

    def test_call_inline_true(self):
        self._test_call_inline(True)

    def test_call_inline_false(self):
        self._test_call_inline(False)


if __name__ == "__main__":
    unittest.main()
