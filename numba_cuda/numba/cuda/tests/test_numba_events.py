import unittest

from numba import cuda
import numba.core.event as ev
from numba.cuda.testing import CUDATestCase


class TestNumbaEvent(CUDATestCase):
    def setUp(self):
        # Trigger compilation to ensure all listeners are initialized
        cuda.jit(lambda: None)[1, 1]()
        self.__registered_listeners = len(ev._registered)

    def tearDown(self):
        # Check there is no lingering listeners
        self.assertEqual(len(ev._registered), self.__registered_listeners)

    def test_compiler_lock_event(self):
        @cuda.jit
        def foo(x):
            _ = x + x

        foo[1, 1](1)
        md = foo.get_metadata(foo.signatures[0])
        lock_duration = md["timers"]["compiler_lock"]
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)

    def test_llvm_lock_event(self):
        @cuda.jit
        def foo(x):
            _ = x + x

        foo[1, 1](1)
        md = foo.get_metadata(foo.signatures[0])
        lock_duration = md["timers"]["llvm_lock"]
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)

    def test_compiler_lock_event_device(self):
        @cuda.jit(device=True)
        def bar(x):
            _ = x + x

        @cuda.jit
        def foo(x):
            bar(x)

        foo[1, 1](1)
        md = bar.get_metadata(bar.signatures[0])
        lock_duration = md["timers"]["compiler_lock"]
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)

    def test_llvm_lock_event_device(self):
        @cuda.jit(device=True)
        def bar(x):
            _ = x + x

        @cuda.jit
        def foo(x):
            bar(x)

        foo[1, 1](1)
        md = bar.get_metadata(bar.signatures[0])
        lock_duration = md["timers"]["llvm_lock"]
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)

    def test_timing_properties(self):
        @cuda.jit(device=True)
        def bar(x):
            _ = x + x

        @cuda.jit
        def foo(x):
            bar(x)

        foo[1, 1](1)

        def get_timers(fn, prop):
            md = fn.get_metadata(fn.signatures[0])
            return md[prop]

        foo_timers = get_timers(foo, "timers")
        bar_timers = get_timers(bar, "timers")

        # Check: time spent in bar() must be less than in foo()
        self.assertLess(bar_timers["llvm_lock"], foo_timers["llvm_lock"])
        self.assertLess(
            bar_timers["compiler_lock"], foo_timers["compiler_lock"]
        )

        # Check: time spent in LLVM lock must be less than in compiler
        self.assertLess(foo_timers["llvm_lock"], foo_timers["compiler_lock"])
        self.assertLess(bar_timers["llvm_lock"], bar_timers["compiler_lock"])


if __name__ == "__main__":
    unittest.main()
