#include "memsys.cuh"

extern "C" __global__ void NRT_MemSys_set(NRT_MemSys *memsys_ptr)
{
  TheMSys = memsys_ptr;
}

extern "C" __global__ void NRT_MemSys_read(uint64_t *managed_memsys)
{
  managed_memsys[0] = TheMSys->stats.alloc;
  managed_memsys[1] = TheMSys->stats.free;
  managed_memsys[2] = TheMSys->stats.mi_alloc;
  managed_memsys[3] = TheMSys->stats.mi_free;
}

extern "C" __global__ void NRT_MemSys_init(void)
{
  TheMSys->stats.enabled = false;
  TheMSys->stats.alloc = 0;
  TheMSys->stats.free = 0;
  TheMSys->stats.mi_alloc = 0;
  TheMSys->stats.mi_free = 0;
}

extern "C" __global__ void NRT_MemSys_enable(void)
{
  TheMSys->stats.enabled = true;
}

extern "C" __global__ void NRT_MemSys_disable(void)
{
  TheMSys->stats.enabled = false;
}

extern "C" __global__ void NRT_MemSys_print(void)
{
  if (TheMSys != nullptr)
  {
    printf("TheMSys->stats.enabled %d\n", TheMSys->stats.enabled);
    printf("TheMSys->stats.alloc %d\n", TheMSys->stats.alloc);
    printf("TheMSys->stats.free %d\n", TheMSys->stats.free);
    printf("TheMSys->stats.mi_alloc %d\n", TheMSys->stats.mi_alloc);
    printf("TheMSys->stats.mi_free %d\n", TheMSys->stats.mi_free);
  } else {
    printf("TheMsys is null.\n");
  }
}