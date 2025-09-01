# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import collections
import numpy as np
import re

from numba.core import types, errors
from numba.cuda.typing.templates import signature
from numba.cuda.np import npdatetime_helpers

numpy_version = tuple(map(int, np.__version__.split(".")[:2]))


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


def select_array_wrapper(inputs):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    select the appropriate input for wrapping the operation output,
    according to each input's __array_priority__.

    An index into *inputs* is returned.
    """
    max_prio = float("-inf")
    selected_index = None
    for index, ty in enumerate(inputs):
        # Ties are broken by choosing the first winner, as in Numpy
        if (
            isinstance(ty, types.ArrayCompatible)
            and ty.array_priority > max_prio
        ):
            selected_index = index
            max_prio = ty.array_priority

    assert selected_index is not None
    return selected_index


def resolve_output_type(context, inputs, formal_output):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    and the operation's formal output type (a types.Array instance),
    resolve the actual output type using the typing *context*.

    This uses a mechanism compatible with Numpy's __array_priority__ /
    __array_wrap__.
    """
    selected_input = inputs[select_array_wrapper(inputs)]
    args = selected_input, formal_output
    sig = context.resolve_function_type("__array_wrap__", args, {})
    if sig is None:
        if selected_input.array_priority == types.Array.array_priority:
            # If it's the same priority as a regular array, assume we
            # should return the output unchanged.
            # (we can't define __array_wrap__ explicitly for types.Buffer,
            #  as that would be inherited by most array-compatible objects)
            return formal_output
        raise errors.TypingError("__array_wrap__ failed for %s" % (args,))
    return sig.return_type


def supported_ufunc_loop(ufunc, loop):
    """Return whether the *loop* for the *ufunc* is supported -in nopython-.

    *loop* should be a UFuncLoopSpec instance, and *ufunc* a numpy ufunc.

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_db
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """
    # NOTE: Assuming ufunc for the CPUContext
    from numba.np import ufunc_db

    loop_sig = loop.ufunc_sig
    try:
        # check if the loop has a codegen description in the
        # ufunc_db. If so, we can proceed.

        # note that as of now not all ufuncs have an entry in the
        # ufunc_db
        supported_loop = loop_sig in ufunc_db.get_ufunc_info(ufunc)
    except KeyError:
        # for ufuncs not in ufunc_db, base the decision of whether the
        # loop is supported on its types
        loop_types = [x.char for x in loop.numpy_inputs + loop.numpy_outputs]
        supported_types = "?bBhHiIlLqQfd"
        # check if all the types involved in the ufunc loop are
        # supported in this mode
        supported_loop = all(t in supported_types for t in loop_types)

    return supported_loop


class UFuncLoopSpec(
    collections.namedtuple("_UFuncLoopSpec", ("inputs", "outputs", "ufunc_sig"))
):
    """
    An object describing a ufunc loop's inner types.  Properties:
    - inputs: the inputs' Numba types
    - outputs: the outputs' Numba types
    - ufunc_sig: the string representing the ufunc's type signature, in
      Numpy format (e.g. "ii->i")
    """

    __slots__ = ()

    @property
    def numpy_inputs(self):
        return [as_dtype(x) for x in self.inputs]

    @property
    def numpy_outputs(self):
        return [as_dtype(x) for x in self.outputs]


def _ufunc_loop_sig(out_tys, in_tys):
    if len(out_tys) == 1:
        return signature(out_tys[0], *in_tys)
    else:
        return signature(types.Tuple(out_tys), *in_tys)


def ufunc_can_cast(from_, to, has_mixed_inputs, casting="safe"):
    """
    A variant of np.can_cast() that can allow casting any integer to
    any real or complex type, in case the operation has mixed-kind
    inputs.

    For example we want `np.power(float32, int32)` to be computed using
    SP arithmetic and return `float32`.
    However, `np.sqrt(int32)` should use DP arithmetic and return `float64`.
    """
    from_ = np.dtype(from_)
    to = np.dtype(to)
    if has_mixed_inputs and from_.kind in "iu" and to.kind in "cf":
        # Decide that all integers can cast to any real or complex type.
        return True
    return np.can_cast(from_, to, casting)


def ufunc_find_matching_loop(ufunc, arg_types):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    arg_types    - The tuple of arguments to the ufunc, including any
                   explicit output(s).
    return value - A UFuncLoopSpec identifying the loop, or None
                   if no matching loop is found.
    """

    # Separate logical input from explicit output arguments
    input_types = arg_types[: ufunc.nin]
    output_types = arg_types[ufunc.nin :]
    assert len(input_types) == ufunc.nin

    try:
        np_input_types = [as_dtype(x) for x in input_types]
    except errors.NumbaNotImplementedError:
        return None
    try:
        np_output_types = [as_dtype(x) for x in output_types]
    except errors.NumbaNotImplementedError:
        return None

    # Whether the inputs are mixed integer / floating-point
    has_mixed_inputs = any(dt.kind in "iu" for dt in np_input_types) and any(
        dt.kind in "cf" for dt in np_input_types
    )

    def choose_types(numba_types, ufunc_letters):
        """
        Return a list of Numba types representing *ufunc_letters*,
        except when the letter designates a datetime64 or timedelta64,
        in which case the type is taken from *numba_types*.
        """
        assert len(ufunc_letters) >= len(numba_types)
        types = [
            tp if letter in "mM" else from_dtype(np.dtype(letter))
            for tp, letter in zip(numba_types, ufunc_letters)
        ]
        # Add missing types (presumably implicit outputs)
        types += [
            from_dtype(np.dtype(letter))
            for letter in ufunc_letters[len(numba_types) :]
        ]
        return types

    def set_output_dt_units(inputs, outputs, ufunc_inputs, ufunc_name):
        """
        Sets the output unit of a datetime type based on the input units

        Timedelta is a special dtype that requires the time unit to be
        specified (day, month, etc). Not every operation with timedelta inputs
        leads to an output of timedelta output. However, for those that do,
        the unit of output must be inferred based on the units of the inputs.

        At the moment this function takes care of two cases:
        a) where all inputs are timedelta with the same unit (mm), and
        therefore the output has the same unit.
        This case is used for arr.sum, and for arr1+arr2 where all arrays
        are timedeltas.
        If in the future this needs to be extended to a case with mixed units,
        the rules should be implemented in `npdatetime_helpers` and called
        from this function to set the correct output unit.
        b) where left operand is a timedelta, i.e. the "m?" case. This case
        is used for division, eg timedelta / int.

        At the time of writing, Numba does not support addition of timedelta
        and other types, so this function does not consider the case "?m",
        i.e. where timedelta is the right operand to a non-timedelta left
        operand. To extend it in the future, just add another elif clause.
        """

        def make_specific(outputs, unit):
            new_outputs = []
            for out in outputs:
                if isinstance(out, types.NPTimedelta) and out.unit == "":
                    new_outputs.append(types.NPTimedelta(unit))
                else:
                    new_outputs.append(out)
            return new_outputs

        def make_datetime_specific(outputs, dt_unit, td_unit):
            new_outputs = []
            for out in outputs:
                if isinstance(out, types.NPDatetime) and out.unit == "":
                    unit = npdatetime_helpers.combine_datetime_timedelta_units(
                        dt_unit, td_unit
                    )
                    if unit is None:
                        raise errors.TypingError(
                            f"ufunc '{ufunc_name}' is not "
                            + "supported between "
                            + f"datetime64[{dt_unit}] "
                            + f"and timedelta64[{td_unit}]"
                        )
                    new_outputs.append(types.NPDatetime(unit))
                else:
                    new_outputs.append(out)
            return new_outputs

        if ufunc_inputs == "mm":
            if all(inp.unit == inputs[0].unit for inp in inputs):
                # Case with operation on same units. Operations on different
                # units not adjusted for now but might need to be
                # added in the future
                unit = inputs[0].unit
                new_outputs = make_specific(outputs, unit)
            else:
                return outputs
            return new_outputs
        elif ufunc_inputs == "mM":
            # case where the left operand has timedelta type
            # and the right operand has datetime
            td_unit = inputs[0].unit
            dt_unit = inputs[1].unit
            return make_datetime_specific(outputs, dt_unit, td_unit)

        elif ufunc_inputs == "Mm":
            # case where the right operand has timedelta type
            # and the left operand has datetime
            dt_unit = inputs[0].unit
            td_unit = inputs[1].unit
            return make_datetime_specific(outputs, dt_unit, td_unit)

        elif ufunc_inputs[0] == "m":
            # case where the left operand has timedelta type
            unit = inputs[0].unit
            new_outputs = make_specific(outputs, unit)
            return new_outputs

    # In NumPy, the loops are evaluated from first to last. The first one
    # that is viable is the one used. One loop is viable if it is possible
    # to cast every input operand to the one expected by the ufunc.
    # Also under NumPy 1.10+ the output must be able to be cast back
    # to a close enough type ("same_kind").

    for candidate in ufunc.types:
        ufunc_inputs = candidate[: ufunc.nin]
        ufunc_outputs = candidate[-ufunc.nout :] if ufunc.nout else []

        if "e" in ufunc_inputs:
            # Skip float16 arrays since we don't have implementation for them
            continue
        if "O" in ufunc_inputs:
            # Skip object arrays
            continue
        found = True
        # Skip if any input or output argument is mismatching
        for outer, inner in zip(np_input_types, ufunc_inputs):
            # (outer is a dtype instance, inner is a type char)
            if outer.char in "mM" or inner in "mM":
                # For datetime64 and timedelta64, we want to retain
                # precise typing (i.e. the units); therefore we look for
                # an exact match.
                if outer.char != inner:
                    found = False
                    break
            elif not ufunc_can_cast(
                outer.char, inner, has_mixed_inputs, "safe"
            ):
                found = False
                break
        if found:
            # Can we cast the inner result to the outer result type?
            for outer, inner in zip(np_output_types, ufunc_outputs):
                if outer.char not in "mM" and not ufunc_can_cast(
                    inner, outer.char, has_mixed_inputs, "same_kind"
                ):
                    found = False
                    break
        if found:
            # Found: determine the Numba types for the loop's inputs and
            # outputs.
            try:
                inputs = choose_types(input_types, ufunc_inputs)
                outputs = choose_types(output_types, ufunc_outputs)
                # if the left operand or both are timedeltas, or the first
                # argument is datetime and the second argument is timedelta,
                # then the output units need to be determined.
                if ufunc_inputs[0] == "m" or ufunc_inputs == "Mm":
                    outputs = set_output_dt_units(
                        inputs, outputs, ufunc_inputs, ufunc.__name__
                    )

            except errors.NumbaNotImplementedError:
                # One of the selected dtypes isn't supported by Numba
                # (e.g. float16), try other candidates
                continue
            else:
                return UFuncLoopSpec(inputs, outputs, candidate)

    return None


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
