#include "fp-histogram.cuh"
#include <cstdarg>
#include "reginfo.cuh"
#include "fp_utils.cuh"
#include "utils/channel.hpp"

using namespace nixnan::fp_histogram;

__device__
void reset_bin_thresholds(BinArray* bins) {
    for (int fmt = 0; fmt < NUM_FORMATS; fmt++) {
        unsigned long long int default_threshold = bins[fmt].default_threshold;
        for (size_t i = 0; i < bins[fmt].num_bins; i++) {
            bins[fmt].bins[i].threshold = default_threshold;
            bins[fmt].bins[i].times_doubled = 0;
        }
    }
}

__device__ __inline__
void record(unsigned long long int* histogram, BinArray* bins,
    int kerid, int format, uint32_t exp, ChannelDev* channel_dev) {
    size_t index = get_index(format, exp);
    atomicAdd(&histogram[index], 1L);
    for (size_t i = 0; i < bins[format].num_bins; i++) {
        BinCounter& bin = bins[format].bins[i];
        unsigned long long int threshold = bin.threshold;
        if (bin.in_bin(exp)) {
            unsigned long long int prev = bin.increment();
            if (prev % threshold == 0 && prev != 0) {
                // Send update to host
                for (auto skip : {false, true}) {
                    //(unsigned long long count, int fmt, int lb, int ub, int kerid, bool to_skip)
                    exp_info ei(threshold, format, bin.lower, bin.upper, kerid, skip);
                    channel_dev->push((void*)&ei, sizeof(ei));
                }
                if (bin.doubling_limit != 0) {
                    if (bin.times_doubled == bin.doubling_limit) {
                        reset_bin_thresholds(bins);
                    } else {
                        bin.threshold *= 2;
                        bin.times_doubled++;
                    }
                }
            }
            break;
        }
    }
}

extern "C" __device__ __noinline__ void
nixnan_fp_histogram_counter(int pred, BinArray* bins,
    unsigned long long int* histogram, ChannelDev* channel_dev, int kerid,
    uint32_t arg_count, ...) {
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
                    record(histogram, bins, kerid, FP16, exp0, channel_dev);
                    uint32_t exp1 = half_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, kerid, FP16, exp1, channel_dev);
                    break;
                }
                case BF16: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    j++;
                    uint32_t exp0 = bf16_exp(val & 0xFFFF);
                    record(histogram, bins, kerid, BF16, exp0, channel_dev);
                    uint32_t exp1 = bf16_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, kerid, BF16, exp1, channel_dev);
                    break;
                }
                case FP32: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = float_exp(val);
                    record(histogram, bins, kerid, FP32, exp, channel_dev);
                    break;
                }
                case FP64: {
                    uint32_t low = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t high = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = double_exp(low, high);
                    record(histogram, bins, kerid, FP64, exp, channel_dev);
                    break;
                }
                default:
                    break;
            }
        }
  }
  va_end(ap);
}