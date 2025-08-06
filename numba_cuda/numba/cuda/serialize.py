"""
Serialization support for compiled functions.
"""

import sys
import io


import pickle
from numba.cuda.cloudpickle import Pickler, loads
from llvmlite import ir


#
# Pickle support
#


def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


# Keep unpickled object via `numba_unpickle` alive.
_unpickled_memo = {}


def _numba_unpickle(address, bytedata, hashed):
    """Used by `numba_unpickle` from _helperlib.c

    Parameters
    ----------
    address : int
    bytedata : bytes
    hashed : bytes

    Returns
    -------
    obj : object
        unpickled object
    """
    key = (address, hashed)
    try:
        obj = _unpickled_memo[key]
    except KeyError:
        _unpickled_memo[key] = obj = loads(bytedata)
    return obj


def dumps(obj):
    """Similar to `pickle.dumps()`. Returns the serialized object in bytes."""
    pickler = Pickler
    with io.BytesIO() as buf:
        p = pickler(buf, protocol=4)
        p.dump(obj)
        pickled = buf.getvalue()

    return pickled


def runtime_build_excinfo_struct(static_exc, exc_args):
    exc, static_args, locinfo = loads(static_exc)
    real_args = []
    exc_args_iter = iter(exc_args)
    for arg in static_args:
        if isinstance(arg, ir.Value):
            real_args.append(next(exc_args_iter))
        else:
            real_args.append(arg)
    return (exc, tuple(real_args), locinfo)


def is_serialiable(obj):
    """Check if *obj* can be serialized.

    Parameters
    ----------
    obj : object

    Returns
    --------
    can_serialize : bool
    """
    with io.BytesIO() as fout:
        pickler = Pickler(fout)
        try:
            pickler.dump(obj)
        except pickle.PicklingError:
            return False
        else:
            return True


def disable_pickling(typ):
    """This is called on a type to disable pickling"""
    Pickler.disabled_types.add(typ)
    # Return `typ` to allow use as a decorator
    return typ


class PickleCallableByPath:
    """Wrap a callable object to be pickled by path to workaround limitation
    in pickling due to non-pickleable objects in function non-locals.

    Note:
    - Do not use this as a decorator.
    - Wrapped object must be a global that exist in its parent module and it
      can be imported by `from the_module import the_object`.

    Usage:

    >>> def my_fn(x):
    >>>     ...
    >>> wrapped_fn = PickleCallableByPath(my_fn)
    >>> # refer to `wrapped_fn` instead of `my_fn`
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __reduce__(self):
        return type(self)._rebuild, (
            self._fn.__module__,
            self._fn.__name__,
        )

    @classmethod
    def _rebuild(cls, modname, fn_path):
        return cls(getattr(sys.modules[modname], fn_path))
