# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.core import sigutils

# Utility functions

# HACK: These are explicitly defined here to avoid having a CExt just to import these constants.
#   np doesn't expose these in the python API.
PyUFunc_Zero = 0
PyUFunc_One = 1
PyUFunc_None = -1
PyUFunc_ReorderableNone = -2


def _compile_element_wise_function(nb_func, targetoptions, sig):
    # Do compilation
    # Return CompileResult to test
    cres = nb_func.compile(sig, **targetoptions)
    args, return_type = sigutils.normalize_signature(sig)
    return cres, args, return_type


# Class definitions


class _BaseUFuncBuilder:
    def add(self, sig=None):
        if hasattr(self, "targetoptions"):
            targetoptions = self.targetoptions
        else:
            targetoptions = self.nb_func.targetoptions
        cres, args, return_type = _compile_element_wise_function(
            self.nb_func, targetoptions, sig
        )
        sig = self._finalize_signature(cres, args, return_type)
        self._sigs.append(sig)
        self._cres[sig] = cres
        return cres

    def disable_compile(self):
        """
        Disable the compilation of new signatures at call time.
        """
        # Override this for implementations that support lazy compilation


_identities = {
    0: PyUFunc_Zero,
    1: PyUFunc_One,
    None: PyUFunc_None,
    "reorderable": PyUFunc_ReorderableNone,
}


def parse_identity(identity):
    """
    Parse an identity value and return the corresponding low-level value
    for Numpy.
    """
    try:
        identity = _identities[identity]
    except KeyError:
        raise ValueError("Invalid identity value %r" % (identity,))
    return identity
