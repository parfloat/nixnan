#include "fp-histogram.cuh"
#include <cstdarg>
#include "reginfo.cuh"
#include "fp_utils.cuh"
#include "utils/channel.hpp"

using namespace nixnan::fp_histogram;

__device__ __inline__
void record(unsigned long long int* histogram, BinArray* bins,
    int kerid, unsigned long count, int format, uint32_t exp,
    ChannelDev* channel_dev, int inst_id) {
    size_t index = get_index(format, exp);
    atomicAdd(&histogram[index], 1L);
    for (size_t i = 0; i < bins[format].num_bins; i++) {
        BinCounter& bin = bins[format].bins[i];
        if (bin.in_bin(exp)) {
            unsigned long long int prev = bin.increment();
            if (prev % count == 0 && prev != 0) {
                // Send update to host
                int reported_inst_id = bin.record_inst ? inst_id : -1;
                for (auto skip : {false, true}) {
                    //(unsigned long long count, int fmt, int lb, int ub, int kerid, int inst_id, bool to_skip)
                    exp_info ei(count, format, bin.lower, bin.upper, kerid, reported_inst_id, skip);
                    channel_dev->push((void*)&ei, sizeof(ei));
                }
            }
            break;
        }
    }
}

extern "C" __device__ __noinline__ void
nixnan_fp_histogram_counter(int pred, BinArray* bins, unsigned long count,
    unsigned long long int* histogram, ChannelDev* channel_dev, int kerid,
    int inst_id, uint32_t arg_count, ...) {
    if (!pred) return;

    va_list ap;
    va_start(ap, arg_count);

    while (arg_count > 0) {
        reginfo reg_info = va_arg(ap, reginfo);
        arg_count--;
        for (int j = 0; j < reg_info.count; j++) {
            auto ritype = reg_info.type;
            switch (ritype) {
                case FP16: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    j++;
                    uint32_t exp0 = half_exp(val & 0xFFFF);
                    record(histogram, bins, kerid, count, FP16, exp0, channel_dev, inst_id);
                    uint32_t exp1 = half_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, kerid, count, FP16, exp1, channel_dev, inst_id);
                    break;
                }
                case BF16: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    j++;
                    uint32_t exp0 = bf16_exp(val & 0xFFFF);
                    record(histogram, bins, kerid, count, BF16, exp0, channel_dev, inst_id);
                    uint32_t exp1 = bf16_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, kerid, count, BF16, exp1, channel_dev, inst_id);
                    break;
                }
                case FP32: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = float_exp(val);
                    record(histogram, bins, kerid, count, FP32, exp, channel_dev, inst_id);
                    break;
                }
                case FP64: {
                    uint32_t low = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t high = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = double_exp(low, high);
                    record(histogram, bins, kerid, count, FP64, exp, channel_dev, inst_id);
                    break;
                }
                default:
                    break;
            }
        }
  }
  va_end(ap);
}