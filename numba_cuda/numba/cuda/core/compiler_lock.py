# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import threading
import functools
import numba.cuda.core.event as ev
from numba.cuda import HAS_NUMBA

if HAS_NUMBA:
    from numba.core.compiler_lock import (
        global_compiler_lock as _numba_compiler_lock,
    )
else:
    _numba_compiler_lock = None


# Lock for the preventing multiple compiler execution
class _CompilerLock(object):
    def __init__(self):
        self._lock = threading.RLock()

    def acquire(self):
        ev.start_event("numba.cuda:compiler_lock")
        self._lock.acquire()

    def release(self):
        self._lock.release()
        ev.end_event("numba.cuda:compiler_lock")

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()

    def is_locked(self):
        is_owned = getattr(self._lock, "_is_owned")
        if not callable(is_owned):
            is_owned = self._is_owned
        return is_owned()

    def __call__(self, func):
        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _acquire_compile_lock

    def _is_owned(self):
        # This method is borrowed from threading.Condition.
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True


_numba_cuda_compiler_lock = _CompilerLock()


# Wrapper that coordinates both numba and numba-cuda compiler locks
class _DualCompilerLock(object):
    """Wrapper that coordinates both the numba-cuda and upstream numba compiler locks."""

    def __init__(self, cuda_lock, numba_lock):
        self._cuda_lock = cuda_lock
        self._numba_lock = numba_lock

    def acquire(self):
        if self._numba_lock:
            self._numba_lock.acquire()
        self._cuda_lock.acquire()

    def release(self):
        self._cuda_lock.release()
        if self._numba_lock:
            self._numba_lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()

    def is_locked(self):
        cuda_locked = self._cuda_lock.is_locked()
        if self._numba_lock:
            return cuda_locked and self._numba_lock.is_locked()
        return cuda_locked

    def __call__(self, func):
        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _acquire_compile_lock


# Create the global compiler lock, wrapping both locks if numba is available
if HAS_NUMBA:
    global_compiler_lock = _DualCompilerLock(
        _numba_cuda_compiler_lock, _numba_compiler_lock
    )
else:
    global_compiler_lock = _numba_cuda_compiler_lock


def require_global_compiler_lock():
    """Sentry that checks the global_compiler_lock is acquired."""
    # Use assert to allow turning off this check
    assert global_compiler_lock.is_locked()
