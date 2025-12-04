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

/*--->
__device__
void report_error(device_recorder recorder, uint32_t inst_id,
                  ChannelDev* pchannel_dev, uint32_t type, uint32_t exce) {
    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    for (int tid = 0; tid < 32; tid++) {
      exce |= __shfl_sync(active_mask, exce, tid);
    }

    // first active lane pushes information on the channel 
    if (exce && first_laneid == laneid) {
        uint32_t num_exceptions = recorder.record(inst_id, E_NAN, 1);
        if (num_exceptions == 0) {
            ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
            for (int skip = 0; skip < 2; skip++) {
                exception_info ei(get_ctaid(), get_warpid(), inst_id, E_NAN, 1, type, skip);
                channel_dev->push(&ei, sizeof(exception_info));
            }
        }
    }
}
<---*/


extern "C" __device__ __noinline__
void my_bar_callback(int tid_x, int tid_y, int tid_z,
                                int ctaid_x, int ctaid_y, int ctaid_z) {
    // Use the coordinates
    // nnout() << "In my_bar_callback" << //printf("Block(%d,%d,%d) Thread(%d,%d,%d)\n" 
    //    ctaid_x, ctaid_y, ctaid_z, tid_x, tid_y, tid_z << std::endl;;

    //printf("Block %d, Thread %d\n", 
    //        blockIdx.x, threadIdx.x);

}

/*
1) bar opcode
2) device channel create in host and pass in: made these
   barChannel_dev
   barChannel_host
3) 

*/