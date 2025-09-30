# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.core import sigutils

# Utility functions


def _compile_element_wise_function(nb_func, targetoptions, sig):
    # Do compilation
    # Return CompileResult to test
    cres = nb_func.compile(sig, **targetoptions)
    args, return_type = sigutils.normalize_signature(sig)
    return cres, args, return_type


# Class definitions


class _BaseUFuncBuilder(object):
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
    0: 0,
    1: 1,
    None: -1,
    "reorderable": -2,
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
