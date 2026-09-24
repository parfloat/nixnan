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
void instrument(CUcontext ctx, Instr* instr, const std::string& kname, CUfunction f);
void term(CUcontext ctx);

class BinCounter {
    public:
    int lower;
    int upper;
    unsigned long long int count;
    bool record_inst;
    // Adaptive threshold doubling (ported from the fp-reset branch): a bin
    // starts reporting every `threshold` occurrences (initially the spec's
    // "count"); each time it reports, threshold doubles, until it has
    // doubled `doubling_limit` times, at which point the NEXT report resets
    // every bin's threshold (in this kernel) back to its starting value.
    // doubling_limit == 0 disables doubling entirely (plain fixed-count
    // reporting, the original behavior).
    unsigned long long int threshold;
    unsigned int times_doubled;
    unsigned int doubling_limit;
    BinCounter(int lower, int upper, unsigned long long int threshold,
               bool record_inst = false, unsigned int doubling_limit = 0)
        : lower(lower), upper(upper), count(0), record_inst(record_inst),
          threshold(threshold), times_doubled(0), doubling_limit(doubling_limit) {}
    BinCounter() : lower(0), upper(0), count(0), record_inst(false),
                   threshold(0), times_doubled(0), doubling_limit(0) {}
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
    unsigned long long int default_threshold;
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
    int inst_id; // -1 unless the triggering bin has record_inst set
    bool _to_skip;

    public:
    __host__ __device__
    exp_info(unsigned long long count, int fmt, int lb, int ub, int kerid, int inst_id, bool to_skip)
        : count(count), fmt(fmt), lb(lb), ub(ub), kerid(kerid), inst_id(inst_id), _to_skip(to_skip) {}

    std::pair<int,int> range() {
        return {lb, ub};
    }

    unsigned long long int get_count() {
        return count;
    }

    int kernel_id() {
        return kerid;
    }

    int instruction_id() {
        return inst_id;
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

#endif // NIXNAN_FP_HISTOGRAM_CUH