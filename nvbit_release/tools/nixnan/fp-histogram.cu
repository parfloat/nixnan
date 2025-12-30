#include "fp-histogram.cuh"
#include "common.cuh"
#include "instruction_info.cuh"
#include "nnout.hh"
#include "utils/channel.hpp"

namespace nixnan {
namespace fp_histogram {

static unsigned long long int* device_histogram = nullptr;
static const size_t num_entries = 4 * (1 << FP64_EXP_BITS);
static BinArray* device_bins = nullptr;

void process_bin_spec() {
    // (cnt, 
    // [ (fmt, [min1, max1], [min2, max2], ... ), 
    //  ... ])

    // example

    // (12,
    // [ (fp32, [2,6], [5,9]),
    //  (fp16, [-4,-2]) ]
    // )
}

void init() {
    GET_VAR_INT(histogram_enabled, "HISTOGRAM", 0, "Enable FP exponent histogramming");
}

int get_exp_bits(uint32_t type) {
    switch (type) {
        case FP16:
            return FP16_EXP_BITS;
        case FP32:
            return FP32_EXP_BITS;
        case FP64:
            return FP64_EXP_BITS;
        case BF16:
            return BF16_EXP_BITS;
        default:
            return -1; // invalid type
    }
}

int get_exp_bias(uint32_t type) {
    int exp_bits = get_exp_bits(type);
    return (1 << (exp_bits - 1)) - 1;
}

void tool_init(CUcontext ctx) {
if (histogram_enabled) {
    cudaMalloc(&device_histogram, num_entries * sizeof(unsigned long long int));
    cudaMemset(device_histogram, 0, num_entries * sizeof(unsigned long long int));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating device histogram: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    BinArray host_bins[NUM_FORMATS];
    cudaMalloc(&device_bins, NUM_FORMATS * sizeof(BinArray));
    for (auto fmt : {BF16, FP16, FP32, FP64}) {
        BinCounter h_cnts[] = {BinCounter(-1,0), BinCounter(2,3), BinCounter(4,5), BinCounter(6,7), BinCounter(10, 16)};
        BinCounter* d_cnts;
        cudaMalloc(&d_cnts, sizeof(h_cnts));
        cudaMemcpy(d_cnts, h_cnts, sizeof(h_cnts), cudaMemcpyHostToDevice);
        host_bins[fmt].bins = d_cnts;
        host_bins[fmt].num_bins = sizeof(h_cnts) / sizeof(BinCounter);
    }
    cudaMemcpy(device_bins, host_bins, NUM_FORMATS * sizeof(BinArray), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating device histogram: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
}

void instrument(CUcontext ctx, Instr* instr) {
if (histogram_enabled) {
    assert(device_histogram != nullptr && "Histogram not initialized!");
    auto reg_infos = instruction_info::get_reginfo(instr);
    if (reg_infos.size() == 0) {
        return;
    }
    nvbit_insert_call(instr, "nixnan_fp_histogram_counter", IPOINT_AFTER);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_bins), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(0ULL), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_histogram), false);
    if (verbose) {
        nnout() << "Histogram instrumenting: " << instr->getSass() << std::endl;
    }
    {
        auto [ri, rfuns] = reg_infos[0];
        nvbit_add_call_arg_const_val32(instr, 1 + rfuns.size());
        nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
        for (auto& rfun : rfuns) {
            rfun();
        }
    }

    nvbit_insert_call(instr, "nixnan_fp_histogram_counter", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_bins), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(0ULL), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_histogram), false);
    size_t num_regs = 0;
    for (size_t i = 1; i < reg_infos.size(); ++i) {
        auto [ri, rfuns] = reg_infos[i];
        num_regs += ri.num_regs;
    }
    // This is the number of registers that were sent as arguments, plus the
    // number of reg_info functions, minus the first one.
    nvbit_add_call_arg_const_val32(instr, num_regs + reg_infos.size() - 1);
    for (size_t i = 1; i < reg_infos.size(); ++i) {
        auto [ri, rfuns] = reg_infos[i];
        nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
        for (auto& rfun : rfuns) {
            rfun();
        }
    }
}
}

std::string exp_with_bias(char format, int exp) {
    if (exp == 0) {
        return "zero";
    }
    if (exp == (1 << get_exp_bits(format)) - 1) {
        return "inf";
    }
    return std::to_string(exp - get_exp_bias(format));
}

void term(CUcontext ctx) {
if (histogram_enabled) {
    unsigned long long int* host_histogram = new unsigned long long int[num_entries];
    cudaMemcpy(host_histogram, device_histogram, num_entries * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        nnout() << "Error copying device histogram to host: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
    bool printed_header = false;
    for (auto format : {BF16, FP16, FP32, FP64}) {
        int min = INT_MAX, max = 0;
        bool seen = false;
        for (int exp = 0; exp < (1 << get_exp_bits(format)); exp++) {
            unsigned long long int count = host_histogram[get_index(format, exp)];
            if (count > 0) {
                seen = true;
                min = std::min(min, exp);
                max = std::max(max, exp);
            }
        }
        if (seen) {
            if (!printed_header) {
                std::cout << "\n";
                nnout() << "--- FP exponent ranges --- \n";
                printed_header = true;
            }
            nnout() << "Exponent range for " << type_to_string.at(format) <<
                       ": [" << exp_with_bias(format, min) << ", " <<
                       exp_with_bias(format, max) << "]\n";
        }
    }
    delete[] host_histogram;
    cudaFree(device_histogram);
    {
        BinArray host_bins[NUM_FORMATS];
        cudaMemcpy(host_bins, device_bins, NUM_FORMATS * sizeof(BinArray), cudaMemcpyDeviceToHost);
        for (auto fmt : {BF16, FP16, FP32, FP64}) {
            BinCounter* d_cnts = host_bins[fmt].bins;
            cudaFree(d_cnts);
        }
        cudaFree(device_bins);
    }
}
}

} // namespace fp_histogram
} // namespace nixnan