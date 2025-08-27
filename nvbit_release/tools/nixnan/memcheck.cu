#include "fp_utils.cuh"
#include <cstdint>
#include "utils/utils.h"
#include <cstdarg>
#include "exception_info.cuh"
#include "recording.h"
#include "utils/channel.hpp"
#include "common.cuh"

using namespace nixnan;

__device__
void report_error(device_recorder recorder, uint32_t inst_id,
                  ChannelDev* pchannel_dev, uint32_t type, uint32_t exce) {
    if (!exce) { return; }
    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        uint32_t num_exceptions = recorder.record(inst_id, E_NAN, 1);
        if (num_exceptions == 0) {
            ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
            exception_info ei(get_ctaid(), get_warpid(), inst_id, E_NAN, 1);
            channel_dev->push(&ei, sizeof(exception_info));
        }
    }
}

extern "C" __device__ __noinline__
void nixnan_check_nans(int pred, device_recorder recorder, uint32_t inst_id,
                  ChannelDev* pchannel_dev, uint32_t type, uint32_t arg_count, ...) {
    if (!pred) { return; }

    va_list args;
    va_start(args, arg_count);
    uint32_t data[2];
    data[0] = va_arg(args, uint32_t);

    if (arg_count == 2) {
        data[1] = va_arg(args, uint32_t);
        if (type == FP64 || type == UNKNOWN) {
            report_error(recorder, inst_id, pchannel_dev, FP64, double_is_nan(data[0], data[1]));
        }
    }

    for (uint32_t i = 0; i < arg_count; i++) {
        uint32_t x = data[i];
        if (type == FP32 || type == UNKNOWN) {
            report_error(recorder, inst_id, pchannel_dev, FP32, float_is_nan(x));
        }
        if (type == FP16 || type == UNKNOWN) {
            report_error(recorder, inst_id, pchannel_dev, FP16, half2_is_nan(x, true, true));
        }
        if (type == BF16 || type == UNKNOWN) {
            report_error(recorder, inst_id, pchannel_dev, BF16, bf162_is_nan(x, true, true));
        }
    }
    va_end(args);
}