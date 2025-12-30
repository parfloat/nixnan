#include "fp-histogram.cuh"
#include <cstdarg>
#include "reginfo.cuh"
#include "fp_utils.cuh"
using nixnan::fp_histogram::get_index;
using nixnan::fp_histogram::BinArray;
using nixnan:: fp_histogram::BinCounter;

__device__ __inline__
void record(unsigned long long int* histogram, BinArray* bins, unsigned long count, int format, uint32_t exp) {
    size_t index = get_index(format, exp);
    atomicAdd(&histogram[index], 1L);
    for (auto fmt : {FP16, BF16, FP32, FP64}) {
        for (size_t i = 0; i < bins->num_bins; i++) {
            BinCounter& bin = bins->bins[i];
            if (bin.in_bin(exp)) {
                bin.increment();
            }
        }
    }
}

extern "C" __device__ __noinline__ void
nixnan_fp_histogram_counter(int pred, BinArray* bins, unsigned long count, unsigned long long int* histogram, uint32_t arg_count, ...) {
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
                    record(histogram, bins, count, FP16, exp0);
                    uint32_t exp1 = half_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, count, FP16, exp1);
                    break;
                }
                case BF16: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    j++;
                    uint32_t exp0 = bf16_exp(val & 0xFFFF);
                    record(histogram, bins, count, BF16, exp0);
                    uint32_t exp1 = bf16_exp((val >> 16) & 0xFFFF);
                    record(histogram, bins, count, BF16, exp1);
                    break;
                }
                case FP32: {
                    uint32_t val = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = float_exp(val);
                    record(histogram, bins, count, FP32, exp);
                    break;
                }
                case FP64: {
                    uint32_t low = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t high = va_arg(ap, uint32_t);
                    arg_count--;
                    uint32_t exp = double_exp(low, high);
                    record(histogram, bins, count, FP64, exp);
                    break;
                }
                default:
                    break;
            }
        }
  }
  va_end(ap);
}