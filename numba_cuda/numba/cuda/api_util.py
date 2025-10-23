# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import functools


def prepare_shape_strides_dtype(shape, strides, dtype, order):
    dtype = np.dtype(dtype)
    if isinstance(shape, (float, np.floating)):
        raise TypeError("shape must be an integer or tuple of integers")
    if isinstance(shape, np.ndarray) and np.issubdtype(
        shape.dtype, np.floating
    ):
        raise TypeError("shape must be an integer or tuple of integers")
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    if isinstance(strides, int):
        strides = (strides,)
    else:
        if not strides:
            strides = _fill_stride_by_order(shape, dtype, order)
        else:
            strides = tuple(strides)
    return shape, strides, dtype


@functools.cache
def _fill_stride_by_order(shape, dtype, order):
    ndims = len(shape)
    if not ndims:
        return ()
    strides = [0] * ndims
    if order == "C":
        strides[-1] = dtype.itemsize
        for d in reversed(range(ndims - 1)):
            strides[d] = strides[d + 1] * shape[d + 1]
    elif order == "F":
        strides[0] = dtype.itemsize
        for d in range(1, ndims):
            strides[d] = strides[d - 1] * shape[d - 1]
    else:
        raise ValueError("must be either C/F order")
    return tuple(strides)
