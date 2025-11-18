/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cooperative_groups.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;

__device__ void _wait_on_tile(cuda::barrier<cuda::thread_scope_block> &tile)
{
    auto token = tile.arrive();
    tile.wait(std::move(token));
}

extern "C"
__device__ int cta_barrier(int *ret) {
    auto cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }

    _wait_on_tile(barrier);
    return 0;
}
