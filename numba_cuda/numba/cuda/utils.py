# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
import warnings
import traceback
import functools

import atexit
import builtins
import importlib
import inspect
import operator
import timeit
import math
import sys
import weakref
import threading
import contextlib
import json
from pprint import pformat

from types import ModuleType
from importlib import import_module
from importlib.util import find_spec
import numpy as np

from inspect import signature as pysignature  # noqa: F401
from inspect import Signature as pySignature  # noqa: F401
from inspect import Parameter as pyParameter  # noqa: F401

from numba.cuda.core.config import (
    MACHINE_BITS,  # noqa: F401
    DEVELOPER_MODE,
)  # noqa: F401

from numba.cuda.core import config

from collections.abc import Sequence

PYVERSION = config.PYVERSION


def erase_traceback(exc_value):
    """
    Erase the traceback and hanging locals from the given exception instance.
    """
    if exc_value.__traceback__ is not None:
        traceback.clear_frames(exc_value.__traceback__)
    return exc_value.with_traceback(None)


def safe_relpath(path, start=os.curdir):
    """
    Produces a "safe" relative path, on windows relpath doesn't work across
    drives as technically they don't share the same root.
    See: https://bugs.python.org/issue7195 for details.
    """
    # find the drive letters for path and start and if they are not the same
    # then don't use relpath!
    drive_letter = lambda x: os.path.splitdrive(os.path.abspath(x))[0]
    drive_path = drive_letter(path)
    drive_start = drive_letter(start)
    if drive_path != drive_start:
        return os.path.abspath(path)
    else:
        return os.path.relpath(path, start=start)


# Mapping between operator module functions and the corresponding built-in
# operators.

BINOPS_TO_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "is": operator.is_,
    "is not": operator.is_not,
    # This one has its args reversed!
    "in": operator.contains,
    "@": operator.matmul,
}

INPLACE_BINOPS_TO_OPERATORS = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "//=": operator.ifloordiv,
    "/=": operator.itruediv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "@=": operator.imatmul,
}


ALL_BINOPS_TO_OPERATORS = {**BINOPS_TO_OPERATORS, **INPLACE_BINOPS_TO_OPERATORS}


UNARY_BUILTINS_TO_OPERATORS = {
    "+": operator.pos,
    "-": operator.neg,
    "~": operator.invert,
    "not": operator.not_,
    "is_true": operator.truth,
}

OPERATORS_TO_BUILTINS = {
    operator.add: "+",
    operator.iadd: "+=",
    operator.sub: "-",
    operator.isub: "-=",
    operator.mul: "*",
    operator.imul: "*=",
    operator.floordiv: "//",
    operator.ifloordiv: "//=",
    operator.truediv: "/",
    operator.itruediv: "/=",
    operator.mod: "%",
    operator.imod: "%=",
    operator.pow: "**",
    operator.ipow: "**=",
    operator.and_: "&",
    operator.iand: "&=",
    operator.or_: "|",
    operator.ior: "|=",
    operator.xor: "^",
    operator.ixor: "^=",
    operator.lshift: "<<",
    operator.ilshift: "<<=",
    operator.rshift: ">>",
    operator.irshift: ">>=",
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.is_: "is",
    operator.is_not: "is not",
    # This one has its args reversed!
    operator.contains: "in",
    # Unary
    operator.pos: "+",
    operator.neg: "-",
    operator.invert: "~",
    operator.not_: "not",
    operator.truth: "is_true",
}


_shutting_down = False


def _at_shutdown():
    global _shutting_down
    _shutting_down = True


def shutting_down(globals=globals):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
    # At shutdown, the attribute may have been cleared or set to None.
    v = globals().get("_shutting_down")
    return v is True or v is None


# weakref.finalize registers an exit function that runs all finalizers for
# which atexit is True. Some of these finalizers may call shutting_down() to
# check whether the interpreter is shutting down. For this to behave correctly,
# we need to make sure that _at_shutdown is called before the finalizer exit
# function. Since atexit operates as a LIFO stack, we first construct a dummy
# finalizer then register atexit to ensure this ordering.
weakref.finalize(lambda: None, lambda: None)
atexit.register(_at_shutdown)


class ThreadLocalStack:
    """A TLS stack container.

    Uses the BORG pattern and stores states in threadlocal storage.
    """

    _tls = threading.local()
    stack_name: str
    _registered = {}

    def __init_subclass__(cls, *, stack_name, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register stack_name mapping to the new subclass
        assert stack_name not in cls._registered, (
            f"stack_name: '{stack_name}' already in use"
        )
        cls.stack_name = stack_name
        cls._registered[stack_name] = cls

    def __init__(self):
        # This class must not be used directly.
        assert type(self) is not ThreadLocalStack
        tls = self._tls
        attr = f"stack_{self.stack_name}"
        try:
            tls_stack = getattr(tls, attr)
        except AttributeError:
            tls_stack = list()
            setattr(tls, attr, tls_stack)

        self._stack = tls_stack

    def push(self, state):
        """Push to the stack"""
        self._stack.append(state)

    def pop(self):
        """Pop from the stack"""
        return self._stack.pop()

    def top(self):
        """Get the top item on the stack.

        Raises IndexError if the stack is empty. Users should check the size
        of the stack beforehand.
        """
        return self._stack[-1]

    def __len__(self):
        return len(self._stack)

    @contextlib.contextmanager
    def enter(self, state):
        """A contextmanager that pushes ``state`` for the duration of the
        context.
        """
        self.push(state)
        try:
            yield
        finally:
            self.pop()


class ConfigOptions(object):
    OPTIONS = {}

    def __init__(self):
        self._values = self.OPTIONS.copy()

    def set(self, name, value=True):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        self._values[name] = value

    def unset(self, name):
        self.set(name, False)

    def _check_attr(self, name):
        if name not in self.OPTIONS:
            raise AttributeError("Invalid flag: %s" % name)

    def __getattr__(self, name):
        self._check_attr(name)
        return self._values[name]

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super(ConfigOptions, self).__setattr__(name, value)
        else:
            self._check_attr(name)
            self._values[name] = value

    def __repr__(self):
        return "Flags(%s)" % ", ".join(
            "%s=%s" % (k, v) for k, v in self._values.items() if v is not False
        )

    def copy(self):
        copy = type(self)()
        copy._values = self._values.copy()
        return copy

    def __eq__(self, other):
        return (
            isinstance(other, ConfigOptions) and other._values == self._values
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self._values.items())))


def order_by_target_specificity(templates, fnkey=""):
    """This orders the given templates from most to least specific against the
    current "target". "fnkey" is an indicative typing key for use in the
    exception message in the case that there's no usable templates for the
    current "target".
    """
    # No templates... return early!
    if templates == []:
        return []

    # fish out templates that are specific to the target if a target is
    # specified
    DEFAULT_TARGET = "generic"
    usable = []
    for ix, temp_cls in enumerate(templates):
        # ? Need to do something about this next line
        md = getattr(temp_cls, "metadata", {})
        hw = md.get("target", DEFAULT_TARGET)
        if hw is not None:
            if hw in ("generic", "cuda"):
                usable.append((temp_cls, ix))

    # sort templates based on target specificity
    # cuda-specific templates get priority before generic ones
    def key(x):
        md = getattr(x[0], "metadata", {})
        hw = md.get("target", DEFAULT_TARGET)
        return (0 if hw == "cuda" else 1, x[1])

    order = [x[0] for x in sorted(usable, key=key)]

    if not order:
        msg = (
            f"Function resolution cannot find any matches for function "
            f"'{fnkey}'."
        )
        from numba.cuda.core.errors import UnsupportedError

        raise UnsupportedError(msg)

    return order


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise AssertionError("key already in dictionary: %r" % (key,))
        super(UniqueDict, self).__setitem__(key, value)


def runonce(fn):
    @functools.wraps(fn)
    def inner():
        if not inner._ran:
            res = fn()
            inner._result = res
            inner._ran = True
        return inner._result

    inner._ran = False
    return inner


def bit_length(intval):
    """
    Return the number of bits necessary to represent integer `intval`.
    """
    assert isinstance(intval, int)
    if intval >= 0:
        return len(bin(intval)) - 2
    else:
        return len(bin(-intval - 1)) - 2


def stream_list(lst):
    """
    Given a list, return an infinite iterator of iterators.
    Each iterator iterates over the list from the last seen point up to
    the current end-of-list.

    In effect, each iterator will give the newly appended elements from the
    previous iterator instantiation time.
    """

    def sublist_iterator(start, stop):
        return iter(lst[start:stop])

    start = 0
    while True:
        stop = len(lst)
        yield sublist_iterator(start, stop)
        start = stop


class BenchmarkResult(object):
    def __init__(self, func, records, loop):
        self.func = func
        self.loop = loop
        self.records = np.array(records) / loop
        self.best = np.min(self.records)

    def __repr__(self):
        name = getattr(self.func, "__name__", self.func)
        args = (name, self.loop, self.records.size, format_time(self.best))
        return "%20s: %10d loops, best of %d: %s per loop" % args


def format_time(tm):
    units = "s ms us ns ps".split()
    base = 1
    for _ in units[:-1]:
        if tm >= base:
            break
        base /= 1000
    else:
        unit = units[-1]
    return "%.1f%s" % (tm / base, unit)


def benchmark(func, maxsec=1):
    timer = timeit.Timer(func)
    number = 1
    result = timer.repeat(1, number)
    # Too fast to be measured
    while min(result) / number == 0:
        number *= 10
        result = timer.repeat(3, number)
    best = min(result) / number
    if best >= maxsec:
        return BenchmarkResult(func, result, number)
        # Scale it up to make it close the maximum time
    max_per_run_time = maxsec / 3 / number
    number = max(max_per_run_time / best / 3, 1)
    # Round to the next power of 10
    number = int(10 ** math.ceil(math.log10(number)))
    records = timer.repeat(3, number)
    return BenchmarkResult(func, records, number)


# A dummy module for dynamically-generated functions
_dynamic_modname = "<dynamic>"
_dynamic_module = ModuleType(_dynamic_modname)
_dynamic_module.__builtins__ = builtins


def chain_exception(new_exc, old_exc):
    """Set the __cause__ attribute on *new_exc* for explicit exception
    chaining.  Returns the inplace modified *new_exc*.
    """
    if DEVELOPER_MODE:
        new_exc.__cause__ = old_exc
    return new_exc


def get_nargs_range(pyfunc):
    """Return the minimal and maximal number of Python function
    positional arguments.
    """
    sig = pysignature(pyfunc)
    min_nargs = 0
    max_nargs = 0
    for p in sig.parameters.values():
        max_nargs += 1
        if p.default == inspect._empty:
            min_nargs += 1
    return min_nargs, max_nargs


def unified_function_type(numba_types, require_precise=True):
    """Returns a unified Numba function type if possible.

    Parameters
    ----------
    numba_types : Sequence of numba Type instances.
    require_precise : bool
      If True, the returned Numba function type must be precise.

    Returns
    -------
    typ : {numba.cuda.types.Type, None}
      A unified Numba function type. Or ``None`` when the Numba types
      cannot be unified, e.g. when the ``numba_types`` contains at
      least two different Numba function type instances.

    If ``numba_types`` contains a Numba dispatcher type, the unified
    Numba function type will be an imprecise ``UndefinedFunctionType``
    instance, or None when ``require_precise=True`` is specified.

    Specifying ``require_precise=False`` enables unifying imprecise
    Numba dispatcher instances when used in tuples or if-then branches
    when the precise Numba function cannot be determined on the first
    occurrence that is not a call expression.
    """
    from numba.cuda.core.errors import NumbaExperimentalFeatureWarning
    from numba.cuda import types

    if not (
        isinstance(numba_types, Sequence)
        and len(numba_types) > 0
        and isinstance(numba_types[0], (types.Dispatcher, types.FunctionType))
    ):
        return

    warnings.warn(
        "First-class function type feature is experimental",
        category=NumbaExperimentalFeatureWarning,
    )

    mnargs, mxargs = None, None
    dispatchers = set()
    function = None
    undefined_function = None

    for t in numba_types:
        if isinstance(t, types.Dispatcher):
            mnargs1, mxargs1 = get_nargs_range(t.dispatcher.py_func)
            if mnargs is None:
                mnargs, mxargs = mnargs1, mxargs1
            elif not (mnargs, mxargs) == (mnargs1, mxargs1):
                return
            dispatchers.add(t.dispatcher)
            t = t.dispatcher.get_function_type()
            if t is None:
                continue
        if isinstance(t, types.FunctionType):
            if mnargs is None:
                mnargs = mxargs = t.nargs
            elif not (mnargs == mxargs == t.nargs):
                return
            if isinstance(t, types.UndefinedFunctionType):
                if undefined_function is None:
                    undefined_function = t
                else:
                    # Refuse to unify using function type
                    return
                dispatchers.update(t.dispatchers)
            else:
                if function is None:
                    function = t
                else:
                    assert function == t
        else:
            return
    if require_precise and (function is None or undefined_function is not None):
        return
    if function is not None:
        if undefined_function is not None:
            assert function.nargs == undefined_function.nargs
            function = undefined_function
    elif undefined_function is not None:
        undefined_function.dispatchers.update(dispatchers)
        function = undefined_function
    else:
        function = types.UndefinedFunctionType(mnargs, dispatchers)

    return function


class _RedirectSubpackage(ModuleType):
    """Redirect a subpackage to a subpackage.

    This allows all references like:

    >>> from numba.old_subpackage import module
    >>> module.item

    >>> import numba.old_subpackage.module
    >>> numba.old_subpackage.module.item

    >>> from numba.old_subpackage.module import item
    """

    def __init__(self, old_module_locals, new_module):
        old_module = old_module_locals["__name__"]
        super().__init__(old_module)

        self.__old_module_states = {}
        self.__new_module = new_module

        new_mod_obj = import_module(new_module)

        # Map all sub-modules over
        for k, v in new_mod_obj.__dict__.items():
            # Get attributes so that `subpackage.xyz` and
            # `from subpackage import xyz` work
            setattr(self, k, v)
            if isinstance(v, ModuleType):
                # Map modules into the interpreter so that
                # `import subpackage.xyz` works
                sys.modules[f"{old_module}.{k}"] = sys.modules[v.__name__]

        # copy across dunders so that package imports work too
        for attr, value in old_module_locals.items():
            if attr.startswith("__") and attr.endswith("__"):
                if attr != "__builtins__":
                    setattr(self, attr, value)
                    self.__old_module_states[attr] = value

    def __reduce__(self):
        args = (self.__old_module_states, self.__new_module)
        return _RedirectSubpackage, args


def redirect_numba_module(old_module_locals, numba_module, numba_cuda_module):
    if find_spec("numba"):
        return _RedirectSubpackage(old_module_locals, numba_module)
    else:
        return _RedirectSubpackage(old_module_locals, numba_cuda_module)


def get_hashable_key(value):
    """
    Given a value, returns a key that can be used
    as a hash. If the value is hashable, we return
    the value, otherwise we return id(value).

    See discussion in gh #6957
    """
    try:
        hash(value)
    except TypeError:
        return id(value)
    else:
        return value


class threadsafe_cached_property(functools.cached_property):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __get__(self, *args, **kwargs):
        with self._lock:
            return super().__get__(*args, **kwargs)


def dump_llvm(fndesc, module):
    print(("LLVM DUMP %s" % fndesc).center(80, "-"))
    if config.HIGHLIGHT_DUMPS:
        try:
            from pygments import highlight
            from pygments.lexers import LlvmLexer as lexer
            from pygments.formatters import Terminal256Formatter
            from numba.cuda.misc.dump_style import by_colorscheme

            print(
                highlight(
                    module.__repr__(),
                    lexer(),
                    Terminal256Formatter(style=by_colorscheme()),
                )
            )
        except ImportError:
            msg = "Please install pygments to see highlighted dumps"
            raise ValueError(msg)
    else:
        print(module)
    print("=" * 80)


class _lazy_pformat(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return pformat(*self.args, **self.kwargs)


class _LazyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _lazy_pformat):
            return str(obj)
        return super().default(obj)


def _readenv(name, ctor, default):
    value = os.environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        if ctor is bool:
            return value.lower() in {"1", "true"}
        return ctor(value)
    except Exception:
        warnings.warn(
            f"Environment variable '{name}' is defined but its associated "
            f"value '{value}' could not be parsed.\n"
            "The parse failed with exception:\n"
            f"{traceback.format_exc()}",
            RuntimeWarning,
        )
        return default


@functools.lru_cache(maxsize=None)
def cached_file_read(filepath, how="r"):
    with open(filepath, how) as f:
        return f.read()


@contextlib.contextmanager
def numba_target_override():
    if importlib.util.find_spec("numba"):
        from numba.core.target_extension import target_override

        with target_override("cuda"):
            yield
    else:
        yield
