# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Utilities for signature normalization and type conversion.

This module also provides type mapping between numba.core types
and numba.cuda types to ensure cross-compatibility.
"""

from numba.cuda import types, typing

try:
    from numba.core.typing import Signature as CoreSignature
    from numba.core import types as core_types

    numba_sig_present = True
    core_types_available = True
except ImportError:
    numba_sig_present = False
    core_types = None
    core_types_available = False


def is_numba_type(ty):
    """
    Check if a type is a numba.core type and not a numba.cuda type.
    """
    if not core_types_available:
        return False
    return isinstance(ty, core_types.Type) and not isinstance(ty, types.Type)


def convert_to_cuda_type(ty):
    """
    Convert a numba.core type to its numba.cuda type equivalent if possible.

    This is the main entry point for type conversion. It handles:
    - numba.core.types -> numba.cuda.types conversion
    - Recursive conversion for container types (arrays, tuples, optionals)
    - Special handling for type wrappers like NumberClass
    - Pass-through for types that are already numba.cuda types
    """
    if not core_types_available:
        return ty

    if isinstance(ty, types.Type):
        return ty

    # If it's not a core type at all, return as-is
    if not isinstance(ty, core_types.Type):
        return ty

    # External types (from third-party libraries) should be returned as-is
    # They have their own typing registrations and shouldn't be converted
    if hasattr(ty, "__module__") and not ty.__module__.startswith("numba."):
        return ty

    if isinstance(ty, core_types.NumberClass):
        cuda_inner = convert_to_cuda_type(ty.instance_type)
        return types.NumberClass(cuda_inner)

    if isinstance(ty, core_types.TypeRef):
        cuda_inner = convert_to_cuda_type(ty.instance_type)
        return types.TypeRef(cuda_inner)

    if isinstance(ty, core_types.Literal):
        return types.literal(ty.literal_value)

    if isinstance(ty, core_types.Record):
        # Convert field types to CUDA types
        cuda_fields = []
        for field_name, field_info in ty.fields.items():
            cuda_field_type = convert_to_cuda_type(field_info.type)
            cuda_fields.append(
                (
                    field_name,
                    {"type": cuda_field_type, "offset": field_info.offset},
                )
            )
        # Create a cuda.types Record with converted field types
        return types.Record(cuda_fields, ty.size, ty.aligned)

    if isinstance(ty, core_types.Array):
        cuda_dtype = convert_to_cuda_type(ty.dtype)
        # Reconstruct array with CUDA dtype, only passing attributes that exist
        kwargs = {}
        if hasattr(ty, "readonly"):
            kwargs["readonly"] = ty.readonly
        if hasattr(ty, "aligned"):
            kwargs["aligned"] = ty.aligned
        return types.Array(cuda_dtype, ty.ndim, ty.layout, **kwargs)

    if isinstance(ty, (core_types.BaseTuple, core_types.BaseAnonymousTuple)):
        cuda_elements = tuple(convert_to_cuda_type(t) for t in ty.types)
        if isinstance(ty, core_types.UniTuple):
            return types.UniTuple(cuda_elements[0], ty.count)
        else:
            return types.Tuple(cuda_elements)

    if isinstance(ty, core_types.Optional):
        cuda_inner = convert_to_cuda_type(ty.type)
        return types.Optional(cuda_inner)

    # Handle simple types via name lookup
    # This includes: Integer, Float, Complex, Boolean, PyObject, etc.
    # Note: Built-in Opaques (none, ellipsis) are converted here
    if hasattr(ty, "name") and hasattr(types, ty.name):
        cuda_type = getattr(types, ty.name)
        if isinstance(cuda_type, types.Type):
            return cuda_type

    # Handle custom Opaque types that didn't match in name lookup above
    # These are user-defined types (e.g., DummyType in numba.cuda.tests)
    if isinstance(ty, core_types.Opaque):
        # Return as-is. User should have appropriate typeof registration
        # for corresponding target (CUDA, cpu)
        return ty

    # Fallback: return as-is (Function, Dispatcher, other special types)
    return ty


def is_signature(sig):
    """
    Return whether *sig* is a potentially valid signature
    specification (for user-facing APIs).
    """
    sig_types = (str, tuple, typing.Signature)
    if numba_sig_present:
        sig_types = (str, tuple, typing.Signature, CoreSignature)
    return isinstance(sig, sig_types)


def _parse_signature_string(signature_str):
    """
    Parameters
    ----------
    signature_str : str
    """
    # Just eval signature_str using the types submodules as globals
    return eval(signature_str, {}, types.__dict__)


def normalize_signature(sig):
    """
    From *sig* (a signature specification), return a ``(args, return_type)``
    tuple, where ``args`` itself is a tuple of types, and ``return_type``
    can be None if not specified.
    """
    if isinstance(sig, str):
        parsed = _parse_signature_string(sig)
    else:
        parsed = sig
    if isinstance(parsed, tuple):
        args, return_type = parsed, None
    else:
        sig_types = (typing.Signature,)
        if numba_sig_present:
            sig_types = (typing.Signature, CoreSignature)
        if isinstance(parsed, sig_types):
            args, return_type = parsed.args, parsed.return_type
        else:
            raise TypeError(
                "invalid signature: %r (type: %r) evaluates to %r "
                "instead of tuple or Signature"
                % (sig, sig.__class__.__name__, parsed.__class__.__name__)
            )

    # Convert core types to CUDA types transparently
    if return_type is not None:
        return_type = convert_to_cuda_type(return_type)
    args = tuple(convert_to_cuda_type(ty) for ty in args)

    def check_type(ty):
        # Accept both CUDA types and numba.core types (for cross-compatibility)
        if not (isinstance(ty, types.Type) or is_numba_type(ty)):
            raise TypeError(
                "invalid type in signature: expected a type "
                "instance, got %r" % (ty,)
            )

    if return_type is not None:
        check_type(return_type)
    for ty in args:
        check_type(ty)

    return args, return_type
