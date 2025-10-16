# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Test extending types via the numba.extending.* API.
"""

from numba.cuda import jit
from numba.cuda import types
from numba.core.errors import TypingError, NumbaTypeError
from numba.cuda.extending import make_attribute_wrapper
from numba.cuda.extending import overload
from numba.cuda.core.imputils import Registry
from numba.cuda.testing import skip_on_cudasim

import unittest

registry = Registry()
lower = registry.lower


def gen_mock_float():
    # Stub to overload, pretending to be `float`. The real `float` function is
    # not used as multiple registrations can collide.
    def mock_float(x):
        pass

    return mock_float


class TestExtTypDummy(unittest.TestCase):
    def setUp(self):
        class DummyType(types.Type):
            def __init__(self):
                super(DummyType, self).__init__(name="Dummy")

        make_attribute_wrapper(DummyType, "value", "value")

        # Store attributes
        self.DummyType = DummyType

    def _add_float_overload(self, mock_float_inst):
        @overload(mock_float_inst)
        def dummy_to_float(x):
            if isinstance(x, self.DummyType):

                def codegen(x):
                    return float(x.value)

                return codegen
            else:
                raise NumbaTypeError("cannot type float({})".format(x))

    @skip_on_cudasim("Simulator does not support extending types")
    def test_overload_float_error_msg(self):
        mock_float = gen_mock_float()
        self._add_float_overload(mock_float)

        @jit
        def foo(x):
            mock_float(x)

        with self.assertRaises(TypingError) as raises:
            foo[1, 1](1j)

        self.assertIn("cannot type float(complex128)", str(raises.exception))
