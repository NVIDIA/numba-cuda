from numba.cuda.testing import (
    skip_on_cudasim,
    skip_unless_cudasim,
    unittest,
    CUDATestCase,
)
from numba import config, cuda

# Basic tests that stream APIs execute on the hardware and in the simulator.
#
# Correctness of semantics is exercised elsewhere in the test suite (though we
# could improve the comprehensiveness of testing by adding more correctness
# tests here in future).


class TestStreamAPI(CUDATestCase):
    def test_stream_create_and_sync(self):
        s = cuda.stream()
        s.synchronize()

    def test_default_stream_create_and_sync(self):
        s = cuda.default_stream()
        s.synchronize()

    def test_legacy_default_stream_create_and_sync(self):
        s = cuda.legacy_default_stream()
        s.synchronize()

    def test_ptd_stream_create_and_sync(self):
        s = cuda.per_thread_default_stream()
        s.synchronize()

    @skip_on_cudasim("External streams are unsupported on the simulator")
    def test_external_stream_create(self):
        #  A dummy pointer value
        ptr = 0x12345678
        s = cuda.external_stream(ptr)
        # We don't test synchronization on the stream because it's not a real
        # stream - we used a dummy pointer for testing the API, so we just
        # ensure that the stream handle matches the external stream pointer.
        if config.CUDA_USE_NVIDIA_BINDING:
            value = int(s.handle)
        else:
            value = s.handle.value
        self.assertEqual(ptr, value)

    @skip_unless_cudasim("External streams are usable with hardware")
    def test_external_stream_simulator_unavailable(self):
        ptr = 0x12345678
        msg = "External streams are unsupported in the simulator"
        with self.assertRaisesRegex(RuntimeError, msg):
            cuda.external_stream(ptr)


if __name__ == "__main__":
    unittest.main()
