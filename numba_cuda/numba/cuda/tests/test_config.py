import numpy as np

from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim

@cuda.jit
def warp_size_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = cuda.config.WARP_SIZE

@cuda.jit
def max_threads_kernel(out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = cuda.config.MAX_THREADS_PER_BLOCK

@cuda.jit
def config_control_flow_kernel(inp, out):
    i = cuda.grid(1)
    if i < inp.size:
        if cuda.config.WARP_SIZE >= 32:
            out[i] = inp[i] * 2
        else:
            out[i] = inp[i]

@skip_on_cudasim("CUDA config values are backend-specific")
class TestCudaConfig(CUDATestCase):

    def _launch_1d(self, kernel, args, size):
        threadsperblock = 128
        blockspergrid = (size + threadsperblock - 1) // threadsperblock
        kernel[blockspergrid, threadsperblock](*args)
        cuda.synchronize()

    def test_warp_size_visible_in_kernel(self):
        out = np.zeros(8, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(warp_size_kernel, (d_out,), out.size)
        result = d_out.copy_to_host()
        # Warp size is expected to be consistent across all threads
        self.assertTrue(np.all(result == result[0]))
        self.assertGreater(result[0], 0)

    def test_max_threads_visible_in_kernel(self):
        out = np.zeros(4, dtype=np.int32)
        d_out = cuda.to_device(out)
        self._launch_1d(max_threads_kernel, (d_out,), out.size)
        result = d_out.copy_to_host()
        self.assertTrue(np.all(result == result[0]))
        self.assertGreaterEqual(result[0], 64)

    def test_config_used_in_control_flow(self):
        inp = np.arange(6, dtype=np.int32)
        out = np.zeros_like(inp)
        d_inp = cuda.to_device(inp)
        d_out = cuda.to_device(out)
        self._launch_1d(
            config_control_flow_kernel,
            (d_inp, d_out),
            inp.size,
        )
        expected = inp * 2 if cuda.config.WARP_SIZE >= 32 else inp
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            expected,
        )
