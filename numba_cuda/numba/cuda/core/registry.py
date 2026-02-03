# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda import utils


class DelayedRegistry(utils.UniqueDict):
    """
    A unique dictionary but with deferred initialisation of the values.

    Attributes
    ----------
    ondemand:

        A dictionary of key -> value, where value is executed
        the first time it is is used.  It is used for part of a deferred
        initialization strategy.
    """

    def __init__(self, *args, **kws):
        self.ondemand = utils.UniqueDict()
        self.key_type = kws.pop("key_type", None)
        self.value_type = kws.pop("value_type", None)
        self._type_check = self.key_type or self.value_type
        super().__init__(*args, **kws)

    def __getitem__(self, item):
        if item in self.ondemand:
            self[item] = self.ondemand[item]()
            del self.ondemand[item]
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if self._type_check:

            def check(x, ty_x):
                if isinstance(ty_x, type):
                    assert ty_x in x.__mro__, (x, ty_x)
                else:
                    assert isinstance(x, ty_x), (x, ty_x)

            if self.key_type is not None:
                check(key, self.key_type)
            if self.value_type is not None:
                check(value, self.value_type)
        return super().__setitem__(key, value)
