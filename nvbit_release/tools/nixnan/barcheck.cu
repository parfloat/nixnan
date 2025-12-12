#include "fp_utils.cuh"
#include <cstdint>
#include "utils/utils.h"
#include <cstdarg>
#include "exception_info.cuh"
#include "recording.h"
#include "utils/channel.hpp"
#include "common.cuh"
//#include "nvbit_reg_rw.h"

using namespace nixnan;

// GPU-side static array to maintain barrier count per block (max 65536 blocks)
__device__ static uint32_t barrier_count_per_block[65536];

// Helper to compute linear block ID from 3D coordinates
__device__ __inline__ uint32_t compute_block_id(int x, int y, int z) {
    return x + y * 256 + z * 256 * 256;  // Assuming max 256 blocks per dimension
}

extern "C" __device__ __noinline__
void my_bar_callback(ChannelDev* pchannel_dev, uint32_t inst_id) {
    // Only thread 0 in each block updates and reports
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        // Compute linear block ID
        uint32_t block_id = compute_block_id(blockIdx.x, blockIdx.y, blockIdx.z);

        // Atomically increment barrier count for this block
        uint32_t barrier_cnt = atomicAdd(&barrier_count_per_block[block_id], 1);

        // Create barrier event info
        int4 cta = make_int4(blockIdx.x, blockIdx.y, blockIdx.z, 0);
        exception_info ei(cta, 0, inst_id, barrier_cnt);

        // Push to channel
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ei, sizeof(exception_info));
    }
}

/*
1) bar opcode
2) device channel create in host and pass in: made these
   barChannel_dev
   barChannel_host
3) 

*/