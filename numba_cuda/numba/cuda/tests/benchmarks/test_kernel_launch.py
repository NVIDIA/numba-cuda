# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import string
from numba import cuda
import numpy as np
import pytest


@pytest.fixture
def many_arrs():
    return [
        cuda.device_array(10000, dtype=np.float32)
        for _ in range(len(string.ascii_lowercase))
    ]


@pytest.fixture
def one_arr():
    return cuda.device_array(10000, dtype=np.float32)


def test_one_arg(benchmark, one_arr):
    @cuda.jit("void(float32[:])")
    def one_arg(arr1):
        return

    benchmark(one_arg[1, 1], one_arr)


def test_many_args(benchmark, many_arrs):
    @cuda.jit("void({})".format(", ".join(["float32[:]"] * len(many_arrs))))
    def many_args(
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        i,
        j,
        k,
        l,
        m,
        n,
        o,
        p,
        q,
        r,
        s,
        t,
        u,
        v,
        w,
        x,
        y,
        z,
    ):
        return

    benchmark(many_args[1, 1], *many_arrs)
