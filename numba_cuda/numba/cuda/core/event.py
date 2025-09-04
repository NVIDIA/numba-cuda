# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
The ``numba.cuda.core.event`` module provides a simple event system for applications
to register callbacks to listen to specific compiler events.

The following events are built in:

- ``"numba:compile"`` is broadcast when a dispatcher is compiling. Events of
  this kind have ``data`` defined to be a ``dict`` with the following
  key-values:

  - ``"dispatcher"``: the dispatcher object that is compiling.
  - ``"args"``: the argument types.
  - ``"return_type"``: the return type.

- ``"numba:compiler_lock"`` is broadcast when the internal compiler-lock is
  acquired. This is mostly used internally to measure time spent with the lock
  acquired.

- ``"numba:llvm_lock"`` is broadcast when the internal LLVM-lock is acquired.
  This is used internally to measure time spent with the lock acquired.

- ``"numba:run_pass"`` is broadcast when a compiler pass is running.

    - ``"name"``: pass name.
    - ``"qualname"``: qualified name of the function being compiled.
    - ``"module"``: module name of the function being compiled.
    - ``"flags"``: compilation flags.
    - ``"args"``: argument types.
    - ``"return_type"`` return type.

Applications can register callbacks that are listening for specific events using
``register(kind: str, listener: Listener)``, where ``listener`` is an instance
of ``Listener`` that defines custom actions on occurrence of the specific event.
"""

import enum
from contextlib import contextmanager, ExitStack
from collections import defaultdict


class EventStatus(enum.Enum):
    """Status of an event."""

    START = enum.auto()
    END = enum.auto()


# Builtin event kinds.
_builtin_kinds = frozenset(
    [
        "numba:compiler_lock",
        "numba:compile",
        "numba:llvm_lock",
        "numba:run_pass",
    ]
)


def _guard_kind(kind):
    """Guard to ensure that an event kind is valid.

    All event kinds with a "numba:" prefix must be defined in the pre-defined
    ``numba.cuda.core.event._builtin_kinds``.
    Custom event kinds are allowed by not using the above prefix.

    Parameters
    ----------
    kind : str

    Return
    ------
    res : str
    """
    if kind.startswith("numba:") and kind not in _builtin_kinds:
        msg = (
            f"{kind} is not a valid event kind, "
            "it starts with the reserved prefix 'numba:'"
        )
        raise ValueError(msg)
    return kind


class Event:
    """An event.

    Parameters
    ----------
    kind : str
    status : EventStatus
    data : any; optional
        Additional data for the event.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``.
    """

    def __init__(self, kind, status, data=None, exc_details=None):
        self._kind = _guard_kind(kind)
        self._status = status
        self._data = data
        self._exc_details = (
            None
            if exc_details is None or exc_details[0] is None
            else exc_details
        )

    @property
    def kind(self):
        """Event kind

        Returns
        -------
        res : str
        """
        return self._kind

    @property
    def status(self):
        """Event status

        Returns
        -------
        res : EventStatus
        """
        return self._status

    @property
    def data(self):
        """Event data

        Returns
        -------
        res : object
        """
        return self._data

    @property
    def is_start(self):
        """Is it a *START* event?

        Returns
        -------
        res : bool
        """
        return self._status == EventStatus.START

    @property
    def is_end(self):
        """Is it an *END* event?

        Returns
        -------
        res : bool
        """
        return self._status == EventStatus.END

    @property
    def is_failed(self):
        """Is the event carrying an exception?

        This is used for *END* event. This method will never return ``True``
        in a *START* event.

        Returns
        -------
        res : bool
        """
        return self._exc_details is None

    def __str__(self):
        data = (
            f"{type(self.data).__qualname__}"
            if self.data is not None
            else "None"
        )
        return f"Event({self._kind}, {self._status}, data: {data})"

    __repr__ = __str__


_registered = defaultdict(list)


def broadcast(event):
    """Broadcast an event to all registered listeners.

    Parameters
    ----------
    event : Event
    """
    for listener in _registered[event.kind]:
        listener.notify(event)


def start_event(kind, data=None):
    """Trigger the start of an event of *kind* with *data*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    evt = Event(kind=kind, status=EventStatus.START, data=data)
    broadcast(evt)


def end_event(kind, data=None, exc_details=None):
    """Trigger the end of an event of *kind*, *exc_details*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``. Or, ``None`` if no error.
    """
    evt = Event(
        kind=kind,
        status=EventStatus.END,
        data=data,
        exc_details=exc_details,
    )
    broadcast(evt)


@contextmanager
def trigger_event(kind, data=None):
    """A context manager to trigger the start and end events of *kind* with
    *data*. The start event is triggered when entering the context.
    The end event is triggered when exiting the context.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    with ExitStack() as scope:

        @scope.push
        def on_exit(*exc_details):
            end_event(kind, data=data, exc_details=exc_details)

        start_event(kind, data=data)
        yield
