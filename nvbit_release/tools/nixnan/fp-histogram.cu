#include "fp-histogram.cuh"
#include "common.cuh"
#include "instruction_info.cuh"
#include "nnout.hh"

namespace nixnan {
namespace fp_histogram {

static unsigned long long int* device_histogram = nullptr;
static const size_t num_entries = 4 * (1 << FP64_EXP_BITS);

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

void tool_init(CUcontext ctx) {
if (histogram_enabled) {
    cudaMalloc(&device_histogram, num_entries * sizeof(unsigned long long int));
    cudaMemset(device_histogram, 0, num_entries * sizeof(unsigned long long int));
    cudaError_t err = cudaGetLastError();
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
    nvbit_insert_call(instr, "nixnan_fp_histogram_counter", IPOINT_AFTER);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_histogram), false);
    if (verbose) {
        nnout() << "Histogram instrumenting instruction with " << 1 + std::get<0>(reg_infos[0]).num_regs << " registers" << std::endl;
    }
    {
        auto [ri, rfuns] = reg_infos[0];
        nvbit_add_call_arg_const_val32(instr, 1 + rfuns.size());
        nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
        if (verbose) {
            nnout() << "Histogram instrumenting: " << instr->getSass() << std::endl;
        }
        for (auto& rfun : rfuns) {
            rfun();
        }
    }

    nvbit_insert_call(instr, "nixnan_fp_histogram_counter", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(instr);
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
        if (verbose) {
        nnout() << "Instrumenting operand " << ri.operand << ". div0: " << ri.div0 << ", regs: " << ri.num_regs << std::endl;
        }
        for (auto& rfun : rfuns) {
        rfun();
        }
    }
}
}

void term(CUcontext ctx) {
if (histogram_enabled) {
    unsigned long long int* host_histogram = new unsigned long long int[num_entries];
    cudaMemcpy(host_histogram, device_histogram, num_entries * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying device histogram to host: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // Print non-zero entries
    for (auto format : {BF16, FP16, FP32, FP64}) {
    }
    delete[] host_histogram;
    cudaFree(device_histogram);
}
}

} // namespace fp_histogram
} // namespace nixnan