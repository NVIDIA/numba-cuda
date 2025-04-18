from numba import cuda
import numba.core.event as ev
from numba.cuda.testing import CUDATestCase


class TestEvent(CUDATestCase):
    def setUp(self):
        # Trigger compilation to ensure all listeners are initialized
        cuda.jit(lambda: None)[1, 1]()
        self.__registered_listeners = len(ev._registered)

    def tearDown(self):
        # Check there is no lingering listeners
        self.assertEqual(len(ev._registered), self.__registered_listeners)

    # def test_recording_listener(self):
    #     @cuda.jit
    #     def foo(x):
    #         r = x + x

    #     with ev.install_recorder("numba:compile") as rec:
    #         foo[1, 1](1)

    #     self.assertIsInstance(rec, ev.RecordingListener)
    #     # Check there must be at least two events.
    #     # Because there must be a START and END for the compilation of foo()
    #     self.assertGreaterEqual(len(rec.buffer), 2)

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
