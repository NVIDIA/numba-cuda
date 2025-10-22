# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from collections import namedtuple
import itertools
import functools
import operator
import numpy as np


Extent = namedtuple("Extent", ["begin", "end"])


def attempt_nocopy_reshape(
    nd, dims, strides, newnd, newdims, newstrides, itemsize, is_f_order
):
    """
    Attempt to reshape an array without copying data.

    This function should correctly handle all reshapes, including
    axes of length 1. Zero strides should work but are untested.

    If a copy is needed, returns 0
    If no copy is needed, returns 1 and fills `newstrides`
    with appropriate strides
    """

    olddims = []
    oldstrides = []
    oldnd = 0

    # Remove axes with dimension 1 from the old array. They have no effect
    # but would need special cases since their strides do not matter.
    for oi in range(nd):
        if dims[oi] != 1:
            olddims.append(dims[oi])
            oldstrides.append(strides[oi])
            oldnd += 1

    # Calculate total sizes
    np_total = 1
    for ni in range(newnd):
        np_total *= newdims[ni]

    op_total = 1
    for oi in range(oldnd):
        op_total *= olddims[oi]

    if np_total != op_total:
        # Different total sizes; no hope
        return 0

    if np_total == 0:
        # Handle zero-sized arrays
        # Just make the strides vaguely reasonable
        # (they can have any value in theory).
        for i in range(newnd):
            newstrides[i] = itemsize
        return 1

    # oi to oj and ni to nj give the axis ranges currently worked with
    oi = 0
    oj = 1
    ni = 0
    nj = 1

    while ni < newnd and oi < oldnd:
        np = newdims[ni]
        op = olddims[oi]

        while np != op:
            if np < op:
                # Misses trailing 1s, these are handled later
                np *= newdims[nj]
                nj += 1
            else:
                op *= olddims[oj]
                oj += 1

        # Check whether the original axes can be combined
        for ok in range(oi, oj - 1):
            if is_f_order:
                if oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]:
                    # not contiguous enough
                    return 0
            else:
                # C order
                if oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]:
                    # not contiguous enough
                    return 0

        # Calculate new strides for all axes currently worked with
        if is_f_order:
            newstrides[ni] = oldstrides[oi]
            for nk in range(ni + 1, nj):
                newstrides[nk] = newstrides[nk - 1] * newdims[nk - 1]
        else:
            # C order
            newstrides[nj - 1] = oldstrides[oj - 1]
            for nk in range(nj - 1, ni, -1):
                newstrides[nk - 1] = newstrides[nk] * newdims[nk]

        ni = nj
        nj += 1
        oi = oj
        oj += 1

    # Set strides corresponding to trailing 1s of the new shape
    if ni >= 1:
        last_stride = newstrides[ni - 1]
    else:
        last_stride = itemsize

    if is_f_order:
        last_stride *= newdims[ni - 1]

    for nk in range(ni, newnd):
        newstrides[nk] = last_stride

    return 1


class Dim(object):
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """

    __slots__ = "start", "stop", "size", "stride", "single"

    def __init__(self, start, stop, size, stride, single):
        self.start = start
        self.stop = stop
        self.size = size
        self.stride = stride
        self.single = single
        assert not single or size == 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.size)
            stride = step * self.stride
            start = self.start + start * abs(self.stride)
            stop = self.start + stop * abs(self.stride)
            if stride == 0:
                size = 1
            else:
                size = _compute_size(start, stop, stride)
            ret = Dim(
                start=start, stop=stop, size=size, stride=stride, single=False
            )
            return ret
        else:
            sliced = self[item : item + 1] if item != -1 else self[-1:]
            if sliced.size != 1:
                raise IndexError
            return Dim(
                start=sliced.start,
                stop=sliced.stop,
                size=sliced.size,
                stride=sliced.stride,
                single=True,
            )

    def get_offset(self, idx):
        return self.start + idx * self.stride

    def __repr__(self):
        strfmt = "Dim(start=%s, stop=%s, size=%s, stride=%s)"
        return strfmt % (self.start, self.stop, self.size, self.stride)

    def normalize(self, base):
        return Dim(
            start=self.start - base,
            stop=self.stop - base,
            size=self.size,
            stride=self.stride,
            single=self.single,
        )

    def copy(self, start=None, stop=None, size=None, stride=None, single=None):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if size is None:
            size = self.size
        if stride is None:
            stride = self.stride
        if single is None:
            single = self.single
        return Dim(start, stop, size, stride, single)

    def is_contiguous(self, itemsize):
        return self.stride == itemsize


def compute_index(indices, dims):
    return sum(d.get_offset(i) for i, d in zip(indices, dims))


class Element(object):
    is_array = False

    def __init__(self, extent):
        self.extent = extent

    def iter_contiguous_extent(self):
        yield self.extent


class Array(object):
    """A dummy numpy array-like object.  Consider it an array without the
    actual data, but offset from the base data pointer.

    Attributes
    ----------
    dims: tuple of Dim
        describing each dimension of the array

    ndim: int
        number of dimension

    shape: tuple of int
        size of each dimension

    strides: tuple of int
        stride of each dimension

    itemsize: int
        itemsize

    extent: (start, end)
        start and end offset containing the memory region
    """

    is_array = True

    @classmethod
    @functools.cache
    def from_desc(cls, offset, shape, strides, itemsize):
        dims = []
        for ashape, astride in zip(shape, strides):
            if not isinstance(ashape, (int, np.integer)):
                raise TypeError("all elements of shape must be ints")
            dim = Dim(
                offset, offset + ashape * astride, ashape, astride, single=False
            )
            dims.append(dim)
            offset = 0  # offset only applies to first dimension
        return cls(dims, itemsize)

    def __init__(self, dims, itemsize):
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.shape = tuple(dim.size for dim in self.dims)
        self.strides = tuple(dim.stride for dim in self.dims)
        self.itemsize = itemsize
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.extent = self._compute_extent()
        self.flags = self._compute_layout()

    def _compute_layout(self):
        # The logic here is based on that in _UpdateContiguousFlags from
        # numpy/core/src/multiarray/flagsobject.c in NumPy v1.19.1 (commit
        # 13661ac70).
        # https://github.com/numpy/numpy/blob/maintenance/1.19.x/numpy/core/src/multiarray/flagsobject.c#L123-L191

        # Records have no dims, and we can treat them as contiguous
        if not self.dims:
            return {"C_CONTIGUOUS": True, "F_CONTIGUOUS": True}

        # If this is a broadcast array then it is not contiguous
        if any([dim.stride == 0 for dim in self.dims]):
            return {"C_CONTIGUOUS": False, "F_CONTIGUOUS": False}

        flags = {"C_CONTIGUOUS": True, "F_CONTIGUOUS": True}

        # Check C contiguity
        sd = self.itemsize
        for dim in reversed(self.dims):
            if dim.size == 0:
                # Contiguous by definition
                return {"C_CONTIGUOUS": True, "F_CONTIGUOUS": True}
            if dim.size != 1:
                if dim.stride != sd:
                    flags["C_CONTIGUOUS"] = False
                sd *= dim.size

        # Check F contiguity
        sd = self.itemsize
        for dim in self.dims:
            if dim.size != 1:
                if dim.stride != sd:
                    flags["F_CONTIGUOUS"] = False
                    return flags
                sd *= dim.size

        return flags

    def _compute_extent(self):
        firstidx = [0] * self.ndim
        lastidx = [s - 1 for s in self.shape]
        start = compute_index(firstidx, self.dims)
        stop = compute_index(lastidx, self.dims) + self.itemsize
        stop = max(stop, start)  # ensure positive extent
        return Extent(start, stop)

    def __repr__(self):
        return "<Array dims=%s itemsize=%s>" % (self.dims, self.itemsize)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = [item]
        else:
            item = list(item)

        nitem = len(item)
        ndim = len(self.dims)
        if nitem > ndim:
            raise IndexError("%d extra indices given" % (nitem - ndim,))

        # Add empty slices for missing indices
        while len(item) < ndim:
            item.append(slice(None, None))

        dims = [dim.__getitem__(it) for dim, it in zip(self.dims, item)]
        newshape = [d.size for d in dims if not d.single]

        arr = Array(dims, self.itemsize)
        if newshape:
            return arr.reshape(*newshape)[0]
        else:
            return Element(arr.extent)

    @property
    def is_c_contig(self):
        return self.flags["C_CONTIGUOUS"]

    @property
    def is_f_contig(self):
        return self.flags["F_CONTIGUOUS"]

    def iter_contiguous_extent(self):
        """Generates extents"""
        if self.is_c_contig or self.is_f_contig:
            yield self.extent
        else:
            if self.dims[0].stride < self.dims[-1].stride:
                innerdim = self.dims[0]
                outerdims = self.dims[1:]
                outershape = self.shape[1:]
            else:
                innerdim = self.dims[-1]
                outerdims = self.dims[:-1]
                outershape = self.shape[:-1]

            if innerdim.is_contiguous(self.itemsize):
                oslen = [range(s) for s in outershape]
                for indices in itertools.product(*oslen):
                    base = compute_index(indices, outerdims)
                    yield base + innerdim.start, base + innerdim.stop
            else:
                oslen = [range(s) for s in self.shape]
                for indices in itertools.product(*oslen):
                    offset = compute_index(indices, self.dims)
                    yield offset, offset + self.itemsize

    def reshape(self, *newdims, **kws):
        oldnd = self.ndim
        newnd = len(newdims)

        if newdims == self.shape:
            return self, None

        order = kws.pop("order", "C")
        if kws:
            raise TypeError("unknown keyword arguments %s" % kws.keys())
        if order not in "CFA":
            raise ValueError("order not C|F|A")

        # check for exactly one instance of -1 in newdims
        # https://github.com/numpy/numpy/blob/623bc1fae1d47df24e7f1e29321d0c0ba2771ce0/numpy/core/src/multiarray/shape.c#L470-L515   # noqa: E501
        unknownidx = -1
        knownsize = 1
        for i, dim in enumerate(newdims):
            if dim < 0:
                if unknownidx == -1:
                    unknownidx = i
                else:
                    raise ValueError("can only specify one unknown dimension")
            else:
                knownsize *= dim

        # compute the missing dimension
        if unknownidx >= 0:
            if knownsize == 0 or self.size % knownsize != 0:
                raise ValueError(
                    "cannot infer valid shape for unknown dimension"
                )
            else:
                newdims = (
                    newdims[0:unknownidx]
                    + (self.size // knownsize,)
                    + newdims[unknownidx + 1 :]
                )

        newsize = functools.reduce(operator.mul, newdims, 1)

        if order == "A":
            order = "F" if self.is_f_contig else "C"

        if newsize != self.size:
            raise ValueError("reshape changes the size of the array")

        if self.is_c_contig or self.is_f_contig:
            if order == "C":
                newstrides = list(iter_strides_c_contig(self, newdims))
            elif order == "F":
                newstrides = list(iter_strides_f_contig(self, newdims))
            else:
                raise AssertionError("unreachable")
        else:
            newstrides = [0] * newnd

            if not attempt_nocopy_reshape(
                oldnd,
                self.shape,
                self.strides,
                newnd,
                newdims,
                newstrides,
                self.itemsize,
                order == "F",
            ):
                raise NotImplementedError("reshape would require copy")

        ret = self.from_desc(
            self.extent.begin,
            shape=newdims,
            strides=newstrides,
            itemsize=self.itemsize,
        )

        return ret, list(self.iter_contiguous_extent())

    def squeeze(self, axis=None):
        newshape, newstrides = [], []
        if axis is None:
            for length, stride in zip(self.shape, self.strides):
                if length != 1:
                    newshape.append(length)
                    newstrides.append(stride)
        else:
            if not isinstance(axis, tuple):
                axis = (axis,)
            for ax in axis:
                if self.shape[ax] != 1:
                    raise ValueError(
                        "cannot select an axis to squeeze out which has size "
                        "not equal to one"
                    )
            for i, (length, stride) in enumerate(zip(self.shape, self.strides)):
                if i not in axis:
                    newshape.append(length)
                    newstrides.append(stride)
        newarr = self.from_desc(
            self.extent.begin,
            shape=newshape,
            strides=newstrides,
            itemsize=self.itemsize,
        )
        return newarr, list(self.iter_contiguous_extent())

    def ravel(self, order="C"):
        if order not in "CFA":
            raise ValueError("order not C|F|A")

        if (
            order in "CA"
            and self.is_c_contig
            or order in "FA"
            and self.is_f_contig
        ):
            newshape = (self.size,)
            newstrides = (self.itemsize,)
            arr = self.from_desc(
                self.extent.begin, newshape, newstrides, self.itemsize
            )
            return arr, list(self.iter_contiguous_extent())

        else:
            raise NotImplementedError("ravel on non-contiguous array")


def iter_strides_f_contig(arr, shape=None):
    """yields the f-contiguous strides"""
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize
    yield itemsize
    sum = 1
    for s in shape[:-1]:
        sum *= s
        yield sum * itemsize


def iter_strides_c_contig(arr, shape=None):
    """yields the c-contiguous strides"""
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize

    def gen():
        yield itemsize
        sum = 1
        for s in reversed(shape[1:]):
            sum *= s
            yield sum * itemsize

    for i in reversed(list(gen())):
        yield i


def is_element_indexing(item, ndim):
    if isinstance(item, slice):
        return False

    elif isinstance(item, tuple):
        if len(item) == ndim:
            if not any(isinstance(it, slice) for it in item):
                return True

    else:
        return True

    return False


def _compute_size(start, stop, step):
    """Algorithm adapted from cpython rangeobject.c"""
    if step > 0:
        lo = start
        hi = stop
    else:
        lo = stop
        hi = start
        step = -step
    if lo >= hi:
        return 0
    return (hi - lo - 1) // step + 1
