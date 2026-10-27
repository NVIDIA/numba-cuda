# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Regression test for CallStack.register orphan-lock defect.

If an exception fires after `CallStack._lock` is acquired but before
the surrounding `try/finally` is entered (e.g. raised by
`CallFrame.__init__`, an OOM, or an asynchronously-injected interrupt),
the lock must still be released.  A bug in the previous implementation
called `self._lock.acquire()` outside the try-block, so any such
exception would orphan the lock and cause subsequent registrations on
the same `CallStack` to block forever.
"""

import threading
import unittest
from unittest import mock

from numba.cuda.typing import context as typing_context


class _BoomError(RuntimeError):
    pass


class _FakeFuncId:
    """Minimal stand-in for FunctionIdentity used by CallStack.register."""

    def __init__(self):
        # `func` only needs to be something `self.match` can compare by
        # identity.  A bare object suffices: an empty CallStack will not
        # match anything regardless.
        self.func = object()


class TestCallStackRegisterOrphanLock(unittest.TestCase):
    def test_lock_released_when_callframe_construction_raises(self):
        """register() must release its lock if CallFrame() raises."""
        stack = typing_context.CallStack()
        func_id = _FakeFuncId()

        with mock.patch.object(
            typing_context,
            "CallFrame",
            side_effect=_BoomError("simulated failure"),
        ):
            with self.assertRaises(_BoomError):
                with stack.register(
                    target=None,
                    typeinfer=None,
                    func_id=func_id,
                    args=(),
                ):
                    # Should never enter the body: CallFrame() raised first.
                    self.fail("register body should not have executed")

        # The lock must be free.  Use a non-blocking acquire with a
        # background thread to prove no thread holds it (RLock would
        # let the same thread re-enter, masking the bug).
        acquired = [False]

        def _try_acquire():
            acquired[0] = stack._lock.acquire(blocking=False)
            if acquired[0]:
                stack._lock.release()

        thread = threading.Thread(target=_try_acquire)
        thread.start()
        thread.join(timeout=5.0)
        self.assertFalse(
            thread.is_alive(), "background thread blocked on orphaned lock"
        )
        self.assertTrue(
            acquired[0],
            "CallStack._lock was orphaned after CallFrame() raised",
        )
        # Stack must remain empty since the append never happened.
        self.assertEqual(len(stack), 0)

    def test_lock_released_on_normal_exit(self):
        """Sanity check: happy path still releases the lock and pops."""
        stack = typing_context.CallStack()
        func_id = _FakeFuncId()

        with mock.patch.object(
            typing_context, "CallFrame", autospec=False
        ) as fake_frame:
            fake_frame.return_value = mock.MagicMock(func_id=func_id, args=())
            with stack.register(
                target=None,
                typeinfer=None,
                func_id=func_id,
                args=(),
            ):
                self.assertEqual(len(stack), 1)

        self.assertEqual(len(stack), 0)
        # Lock free in a fresh thread.
        acquired = [False]

        def _try_acquire():
            acquired[0] = stack._lock.acquire(blocking=False)
            if acquired[0]:
                stack._lock.release()

        thread = threading.Thread(target=_try_acquire)
        thread.start()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired[0])


if __name__ == "__main__":
    unittest.main()
