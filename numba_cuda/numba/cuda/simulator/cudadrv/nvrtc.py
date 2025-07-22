"""
NVVM is not supported in the simulator, but stubs are provided to allow tests
to import correctly.
"""


def compile(src, name, cc, ltoir=False):
    raise RuntimeError("NVRTC is not supported in the simulator")
