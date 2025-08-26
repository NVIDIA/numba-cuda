# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import re
from numba.core import types, errors, config


numpy_version = tuple(map(int, np.__version__.split(".")[:2]))


if getattr(config, "USE_LEGACY_TYPE_SYSTEM", True):
    FROM_DTYPE = {
        np.dtype("bool"): types.boolean,
        np.dtype("int8"): types.int8,
        np.dtype("int16"): types.int16,
        np.dtype("int32"): types.int32,
        np.dtype("int64"): types.int64,
        np.dtype("uint8"): types.uint8,
        np.dtype("uint16"): types.uint16,
        np.dtype("uint32"): types.uint32,
        np.dtype("uint64"): types.uint64,
        np.dtype("float32"): types.float32,
        np.dtype("float64"): types.float64,
        np.dtype("float16"): types.float16,
        np.dtype("complex64"): types.complex64,
        np.dtype("complex128"): types.complex128,
        np.dtype(object): types.pyobject,
    }
else:
    FROM_DTYPE = {
        np.dtype("bool"): types.np_bool_,
        np.dtype("int8"): types.np_int8,
        np.dtype("int16"): types.np_int16,
        np.dtype("int32"): types.np_int32,
        np.dtype("int64"): types.np_int64,
        np.dtype("uint8"): types.np_uint8,
        np.dtype("uint16"): types.np_uint16,
        np.dtype("uint32"): types.np_uint32,
        np.dtype("uint64"): types.np_uint64,
        np.dtype("float32"): types.np_float32,
        np.dtype("float64"): types.np_float64,
        np.dtype("float16"): types.np_float16,
        np.dtype("complex64"): types.np_complex64,
        np.dtype("complex128"): types.np_complex128,
        np.dtype(object): types.pyobject,
    }


re_typestr = re.compile(r"[<>=\|]([a-z])(\d+)?$", re.I)
re_datetimestr = re.compile(r"[<>=\|]([mM])8?(\[([a-z]+)\])?$", re.I)

sizeof_unicode_char = np.dtype("U1").itemsize


def _from_str_dtype(dtype):
    m = re_typestr.match(dtype.str)
    if not m:
        raise errors.NumbaNotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    if typecode == "U":
        # unicode
        if dtype.byteorder not in "=|":
            raise errors.NumbaNotImplementedError(
                "Does not support non-native byteorder"
            )
        count = dtype.itemsize // sizeof_unicode_char
        assert count == int(groups[1]), "Unicode char size mismatch"
        return types.UnicodeCharSeq(count)

    elif typecode == "S":
        # char
        count = dtype.itemsize
        assert count == int(groups[1]), "Char size mismatch"
        return types.CharSeq(count)

    else:
        raise errors.NumbaNotImplementedError(dtype)


def _from_datetime_dtype(dtype):
    m = re_datetimestr.match(dtype.str)
    if not m:
        raise errors.NumbaNotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    unit = groups[2] or ""
    if typecode == "m":
        return types.NPTimedelta(unit)
    elif typecode == "M":
        return types.NPDatetime(unit)
    else:
        raise errors.NumbaNotImplementedError(dtype)


def from_dtype(dtype):
    """
    Return a Numba Type instance corresponding to the given Numpy *dtype*.
    NumbaNotImplementedError is raised on unsupported Numpy dtypes.
    """
    if type(dtype) is type and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)
    elif getattr(dtype, "fields", None) is not None:
        return from_struct_dtype(dtype)

    try:
        return FROM_DTYPE[dtype]
    except KeyError:
        pass

    try:
        char = dtype.char
    except AttributeError:
        pass
    else:
        if char in "SU":
            return _from_str_dtype(dtype)
        if char in "mM":
            return _from_datetime_dtype(dtype)
        if char in "V" and dtype.subdtype is not None:
            subtype = from_dtype(dtype.subdtype[0])
            return types.NestedArray(subtype, dtype.shape)

    raise errors.NumbaNotImplementedError(dtype)


_as_dtype_letters = {
    types.NPDatetime: "M8",
    types.NPTimedelta: "m8",
    types.CharSeq: "S",
    types.UnicodeCharSeq: "U",
}


def as_struct_dtype(rec):
    """Convert Numba Record type to NumPy structured dtype"""
    assert isinstance(rec, types.Record)
    names = []
    formats = []
    offsets = []
    titles = []
    # Fill the fields if they are not a title.
    for k, t in rec.members:
        if not rec.is_title(k):
            names.append(k)
            formats.append(as_dtype(t))
            offsets.append(rec.offset(k))
            titles.append(rec.fields[k].title)

    fields = {
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": rec.size,
        "titles": titles,
    }
    _check_struct_alignment(rec, fields)
    return np.dtype(fields, align=rec.aligned)


def _check_struct_alignment(rec, fields):
    """Check alignment compatibility with Numpy"""
    if rec.aligned:
        for k, dt in zip(fields["names"], fields["formats"]):
            llvm_align = rec.alignof(k)
            npy_align = dt.alignment
            if llvm_align is not None and npy_align != llvm_align:
                msg = (
                    "NumPy is using a different alignment ({}) "
                    "than Numba/LLVM ({}) for {}. "
                    "This is likely a NumPy bug."
                )
                raise ValueError(msg.format(npy_align, llvm_align, dt))


def as_dtype(nbtype):
    """
    Return a numpy dtype instance corresponding to the given Numba type.
    NotImplementedError is if no correspondence is known.
    """
    nbtype = types.unliteral(nbtype)
    if isinstance(nbtype, (types.Complex, types.Integer, types.Float)):
        return np.dtype(str(nbtype))
    if isinstance(nbtype, (types.Boolean)):
        return np.dtype("?")
    if isinstance(nbtype, (types.NPDatetime, types.NPTimedelta)):
        letter = _as_dtype_letters[type(nbtype)]
        if nbtype.unit:
            return np.dtype("%s[%s]" % (letter, nbtype.unit))
        else:
            return np.dtype(letter)
    if isinstance(nbtype, (types.CharSeq, types.UnicodeCharSeq)):
        letter = _as_dtype_letters[type(nbtype)]
        return np.dtype("%s%d" % (letter, nbtype.count))
    if isinstance(nbtype, types.Record):
        return as_struct_dtype(nbtype)
    if isinstance(nbtype, types.EnumMember):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.npytypes.DType):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NumberClass):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NestedArray):
        spec = (as_dtype(nbtype.dtype), tuple(nbtype.shape))
        return np.dtype(spec)
    if isinstance(nbtype, types.PyObject):
        return np.dtype(object)

    msg = f"{nbtype} cannot be represented as a NumPy dtype"
    raise errors.NumbaNotImplementedError(msg)


def _is_aligned_struct(struct):
    return struct.isalignedstruct


def from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type"""
    if dtype.hasobject:
        msg = "dtypes that contain object are not supported."
        raise errors.NumbaNotImplementedError(msg)

    fields = []
    for name, info in dtype.fields.items():
        # *info* may have 3 element
        [elemdtype, offset] = info[:2]
        title = info[2] if len(info) == 3 else None

        ty = from_dtype(elemdtype)
        infos = {
            "type": ty,
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))

    # Note: dtype.alignment is not consistent.
    #       It is different after passing into a recarray.
    #       recarray(N, dtype=mydtype).dtype.alignment != mydtype.alignment
    size = dtype.itemsize
    aligned = _is_aligned_struct(dtype)

    return types.Record(fields, size, aligned)
