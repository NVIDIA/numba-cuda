from numba import config

rtsys = None

config.CUDA_NRT_STATS = False
config.CUDA_ENABLE_NRT = False
