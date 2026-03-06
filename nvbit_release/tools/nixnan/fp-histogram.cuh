#ifndef NIXNAN_FP_HISTOGRAM_CUH
#define NIXNAN_FP_HISTOGRAM_CUH
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
void instrument(CUcontext ctx, Instr* instr, const std::string& kname);
void term(CUcontext ctx);

class BinCounter {
    public:
    int lower;
    int upper;
    unsigned long long int count;
    BinCounter(int lower, int upper) : lower(lower), upper(upper), count(0) {}
    BinCounter() : lower(0), upper(0), count(0) {}
    __device__
    bool in_bin(int value) {
        return value >= lower && value <= upper;
    }
    __device__
    unsigned long long int increment() {
        return atomicAdd(&count, 1);
    }
};

struct BinArray {
    BinCounter* bins;
    size_t num_bins;
};

__inline__ __host__ __device__
size_t get_index(int format, uint32_t exp) {
    return format << FP64_EXP_BITS | exp;
}

class exp_info {
    unsigned long long count;
    int lb;
    int ub;
    int kerid;
    int fmt;
    bool _to_skip;

    public:
    __host__ __device__
    exp_info(unsigned long long count, int fmt, int lb, int ub, int kerid, bool to_skip)
        : count(count), fmt(fmt), lb(lb), ub(ub), kerid(kerid), _to_skip(to_skip) {}

    std::pair<int,int> range() {
        return {lb, ub};
    }

    unsigned long long int get_count() {
        return count;
    }

    int kernel_id() {
        return kerid;
    }

    int warp() {
        return 0;
    }

    bool to_skip() {
        return _to_skip;
    }

    int format() {
        return fmt;
    }
};

} // namespace fp_histogram
} // namespace nixnan

extern "C" {
    unsigned long long int* nixnan_fp_histogram_get_histogram();
    void nixnan_fp_histogram_destroy_histogram(unsigned long long int* histogram);
}

#endif // NIXNAN_FP_HISTOGRAM_CUH