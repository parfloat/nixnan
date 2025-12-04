#ifndef NIXNAN_FP_HISTOGRAM_CUH__
#define NIXNAN_FP_HISTOGRAM_CUH_
#include "nvbit.h"
namespace nixnan {
namespace fp_histogram {
static const int BF16_EXP_BITS = 8;
static const int FP16_EXP_BITS = 5;
static const int FP32_EXP_BITS = 8;
static const int FP64_EXP_BITS = 11;
static const int BF16_EXP_BIAS = 1<<BF16_EXP_BITS/2 - 1;
static const int FP16_EXP_BIAS = 1<<FP16_EXP_BITS/2 - 1;
static const int FP32_EXP_BIAS = 1<<FP32_EXP_BITS/2 - 1;
static const int FP64_EXP_BIAS = 1<<FP64_EXP_BITS/2 - 1;
static int histogram_enabled = false;

void init();
void tool_init(CUcontext ctx);
void instrument(CUcontext ctx, Instr* instr);
void term(CUcontext ctx);

__inline__ __host__ __device__
size_t get_index(int format, uint32_t exp) {
    return format << FP64_EXP_BITS | exp;
}
} // namespace fp_histogram
} // namespace nixnan

#endif // NIXNAN_FP_HISTOGRAM_CUH__