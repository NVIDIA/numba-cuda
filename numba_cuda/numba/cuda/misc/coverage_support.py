# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Implement code coverage support.

Currently contains logic to extend ``coverage`` with lines covered by the
compiler.
"""

from typing import Optional, Sequence, Callable
from abc import ABC, abstractmethod

from numba.core import ir, config

_the_registry: Callable[[], Optional["NotifyLocBase"]] = []


def get_registered_loc_notify() -> Sequence["NotifyLocBase"]:
    """
    Returns a list of the registered NotifyLocBase instances.
    """
    if not config.JIT_COVERAGE:
        # Coverage disabled.
        return []
    return list(
        filter(
            lambda x: x is not None, (factory() for factory in _the_registry)
        )
    )


class NotifyLocBase(ABC):
    """Interface for notifying visiting of a ``numba.core.ir.Loc``."""

    @abstractmethod
    def notify(self, loc: ir.Loc) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
