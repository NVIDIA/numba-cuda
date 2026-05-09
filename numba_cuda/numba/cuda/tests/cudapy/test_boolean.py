# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba.cuda.testing import CUDATestCase
from numba import cuda


def boolean_func(A, vertial):
    if vertial:
        A[0] = 123
    else:
        A[0] = 321


def test_boolean():
    func = cuda.jit("void(float64[:], bool_)")(boolean_func)
    A = np.array([0], dtype="float64")
    func[1, 1](A, True)
    assert A[0] == 123
    func[1, 1](A, False)
    assert A[0] == 321
