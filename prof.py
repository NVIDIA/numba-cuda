from numba import cuda
import numpy as np

import pyinstrument


@cuda.jit("void(float32[:])")
def some_kernel(arr1):
    return


arr = cuda.device_array(10000, dtype=np.float32)

kern = some_kernel[1, 1]

# burn 10 calls for warmup
for _ in range(10):
    kern(arr)

profiler = pyinstrument.Profiler(interval=0.000001)
profiler.start()
kern(arr)
profiler.stop()
with open("/tmp/out.html", "w") as f:
    f.write(profiler.output_html())
