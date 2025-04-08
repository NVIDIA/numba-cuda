#include "memsys.cuh"

__device__ size_t memsys_size = sizeof(NRT_MemSys);

namespace detail
{
  void __device__ check_memsys()
  {
    if (TheMSys == nullptr)
    {
      assert(false && "TheMSys pointer is null. Please use NRT_MemSys_set to set pointer first.");
    }
  }
}

extern "C" __global__ void NRT_MemSys_set(NRT_MemSys *memsys_ptr)
{
  TheMSys = memsys_ptr;
}

extern "C" __global__ void NRT_MemSys_read(uint64_t *managed_memsys)
{
  detail::check_memsys();
  managed_memsys[0] = TheMSys->stats.alloc;
  managed_memsys[1] = TheMSys->stats.free;
  managed_memsys[2] = TheMSys->stats.mi_alloc;
  managed_memsys[3] = TheMSys->stats.mi_free;
}

extern "C" __global__ void NRT_MemSys_read_alloc(uint64_t *managed_result)
{
  detail::check_memsys();
  managed_result[0] = TheMSys->stats.alloc;
}

extern "C" __global__ void NRT_MemSys_read_free(uint64_t *managed_result)
{
  detail::check_memsys();
  managed_result[0] = TheMSys->stats.free;
}

extern "C" __global__ void NRT_MemSys_read_mi_alloc(uint64_t *managed_result)
{
  detail::check_memsys();
  managed_result[0] = TheMSys->stats.mi_alloc;
}

extern "C" __global__ void NRT_MemSys_read_mi_free(uint64_t *managed_result)
{
  detail::check_memsys();
  managed_result[0] = TheMSys->stats.mi_free;
}

extern "C" __global__ void NRT_MemSys_init(void)
{
  detail::check_memsys();
  TheMSys->stats.enabled = false;
  TheMSys->stats.alloc = 0;
  TheMSys->stats.free = 0;
  TheMSys->stats.mi_alloc = 0;
  TheMSys->stats.mi_free = 0;
}

extern "C" __global__ void NRT_MemSys_enable_stats(void)
{
  detail::check_memsys();
  TheMSys->stats.enabled = true;
}

extern "C" __global__ void NRT_MemSys_disable_stats(void)
{
  detail::check_memsys();
  TheMSys->stats.enabled = false;
}

extern "C" __global__ void NRT_MemSys_stats_enabled(uint8_t *enabled)
{
  detail::check_memsys();
  *enabled = static_cast<uint8_t>(TheMSys->stats.enabled);
}

extern "C" __global__ void NRT_MemSys_print(void)
{
  if (TheMSys != nullptr)
  {
    printf("TheMSys->stats.enabled %d\n", TheMSys->stats.enabled);
    printf("TheMSys->stats.alloc %lu\n", TheMSys->stats.alloc.load());
    printf("TheMSys->stats.free %lu\n", TheMSys->stats.free.load());
    printf("TheMSys->stats.mi_alloc %lu\n", TheMSys->stats.mi_alloc.load());
    printf("TheMSys->stats.mi_free %lu\n", TheMSys->stats.mi_free.load());
  } else {
    printf("TheMsys is null.\n");
  }
}
