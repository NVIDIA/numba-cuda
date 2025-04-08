# from cuda.core.experimental import Program, ProgramOptions

# kernel = """extern "C" __global__ void ABC() { }"""
# object_code = Program(kernel, "c++", options=ProgramOptions(relocatable_device_code=True)).compile("ptx")
# assert object_code._handle is None
# kernel = object_code.get_kernel("ABC")
# assert object_code._handle is not None
# assert kernel._handle is not None


from cuda.bindings.driver import (
    cuInit,
    cuLibraryLoadFromFile,
    cuLibraryLoadData,
    cuLibraryGetModule,
    cuModuleGetGlobal,
)

from cuda.core.experimental import Device, ObjectCode, LaunchConfig, launch

cuInit(0)

d = Device()
d.set_current()
# s = d.create_stream()

data = "./test.cubin"

with open(data, "rb") as f:
    cubin = f.read()

# obj = ObjectCode.from_cubin(cubin)
# kern = obj.get_kernel("_Z6kernelv")
# conf = LaunchConfig(block=1, grid=1)
# launch(s, conf, kern)

res, culib = cuLibraryLoadData(
    cubin,
    [],
    [],
    0,
    [],
    [],
    0
)

print(res, culib)

res, mod = cuLibraryGetModule(culib)

print(res, mod)

res, ptr, sz = cuModuleGetGlobal(mod, "num".encode())

print(res, ptr, sz)

