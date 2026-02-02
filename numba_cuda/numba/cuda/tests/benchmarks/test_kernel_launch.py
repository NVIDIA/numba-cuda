# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import string
from numba import cuda
from numba.cuda.core import config
import numpy as np
import pytest
from pytest import param
from numba.cuda.testing import DeprecatedDeviceArrayApiWarning


pytestmark = pytest.mark.skipif(
    condition=config.ENABLE_CUDASIM,
    reason="no reason to run benchmarks in the simulator",
)

if not config.ENABLE_CUDASIM:
    with pytest.warns(DeprecatedDeviceArrayApiWarning):
        devary_arg = cuda.device_array(128, dtype=np.float32)
else:
    devary_arg = cuda.device_array(128, dtype=np.float32)


@pytest.mark.parametrize(
    "array_func",
    [
        param(
            lambda: devary_arg,
            id="device_array",
        ),
        param(
            lambda: pytest.importorskip("torch").empty(
                (128,),
                dtype=pytest.importorskip("torch").float32,
                device="cuda:0",
            ),
            id="torch",
        ),
        param(
            lambda: pytest.importorskip("cupy").empty(128, dtype=np.float32),
            id="cupy",
        ),
    ],
)
@pytest.mark.parametrize(
    "jit",
    [cuda.jit, cuda.jit("void(float32[::1])")],
    ids=["dispatch", "signature"],
)
def test_one_arg(benchmark, array_func, jit):
    @jit
    def one_arg(arr1):
        return

    benchmark(one_arg[128, 128], array_func())


@pytest.mark.parametrize(
    "array_func",
    [
        param(
            lambda: [devary_arg for _ in range(len(string.ascii_lowercase))],
            id="device_array",
        ),
        param(
            lambda: [
                pytest.importorskip("torch").empty(
                    (128,),
                    dtype=pytest.importorskip("torch").float32,
                    device="cuda:0",
                )
                for _ in range(len(string.ascii_lowercase))
            ],
            id="torch",
        ),
        param(
            lambda: [
                pytest.importorskip("cupy").empty(128, dtype=np.float32)
                for _ in range(len(string.ascii_lowercase))
            ],
            id="cupy",
        ),
    ],
)
@pytest.mark.parametrize(
    "jit",
    [
        cuda.jit,
        cuda.jit(
            "void({})".format(
                ", ".join(["float32[::1]"] * len(string.ascii_lowercase))
            )
        ),
    ],
    ids=["dispatch", "signature"],
)
def test_many_args(benchmark, array_func, jit):
    @jit
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

    benchmark(many_args[128, 128], *array_func())
