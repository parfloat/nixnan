#include "fp-histogram.cuh"
#include "common.cuh"
#include "instruction_info.cuh"
#include "nnout.hh"
#include "utils/channel.hpp"
#include <thread>
#include <atomic>
#include "nlohmann/json.hpp"
#include <fstream>

namespace nixnan {
namespace fp_histogram {

static unsigned long long int* device_histogram = nullptr;
static const size_t num_entries = 4 * (1 << FP64_EXP_BITS);
static BinArray* device_bins = nullptr;
static unsigned long long int count_threshold = 0;

static std::atomic<bool> recv_thread_running;
static std::atomic<bool> recv_thread_receiving;
static ChannelHost channel_host;
static __managed__ ChannelDev  channel_dev;
std::thread recv_thread;
std::string bin_spec_file;
std::unordered_map<std::string, uint32_t> kernel_to_id;
std::unordered_map<uint32_t, std::string> id_to_kernel;
template<typename T>
void recv_thread_fun(std::atomic<bool> *recv_thread_running,
                     std::atomic<bool> *recv_thread_receiving,
                     ChannelHost channel_host,
                     std::function<void(T*)> process_data) {
  size_t CHANNEL_SIZE = sizeof(T);
  char *recv_buffer = new char[CHANNEL_SIZE];

  while (*recv_thread_running) {
    uint32_t num_recv_bytes = 0;

    if (*recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        T *data = reinterpret_cast<T*>(&recv_buffer[num_processed_bytes]);
        if (data->to_skip()) {
          num_processed_bytes += CHANNEL_SIZE;
          continue;
        }
        if (data->warp() == -1) {
          *recv_thread_receiving = false;
          break;
        }
        process_data(data);
        num_processed_bytes += CHANNEL_SIZE;
      }
    }
  }
  delete[] recv_buffer;
  return;
}

void init() {
    GET_VAR_INT(histogram_enabled, "HISTOGRAM", 0, "Enable FP exponent histogramming");
    GET_VAR_STR(bin_spec_file, "BIN_SPEC_FILE",
                R"(Specification for which exponent ranges to report. If the file does not exist, a template specification will be created.
For example:
{
    "count": 128,
    "bf16": [],
    "f16": [[0,5],[-4,-1]],
    "f32": [],
    "f64": []
}
will report every 128 occurrences of exponents in the ranges 0 to 5 and -4 to -1 for f16 numbers.)");
    if (bin_spec_file != "") {
        histogram_enabled = true;
        std::fstream bin_spec_ifs(bin_spec_file);
        if (!bin_spec_ifs.good()) {
            try {
                bin_spec_ifs = std::fstream(bin_spec_file, std::ios::out | std::ios::trunc);
                std::string default_spec = R"({
    "count": 128,
    "bf16": [],
    "f16": [],
    "f32": [],
    "f64": []
})";
                bin_spec_ifs.write(default_spec.c_str(), default_spec.size());
                bin_spec_ifs.close();
                nnout() << "Created template bin specification file at " << bin_spec_file << "\nExiting now. Please edit the file to specify which exponent ranges to report.\n";
                exit(0);
            } catch (...) {
                nnout() << "Error creating template bin specification file at " << bin_spec_file << "\nExiting now.\n";
                exit(1);
            }
        }
    }
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

BinCounter bin_from_json(const nlohmann::json& j, unsigned char fmt) {
    int bias = get_exp_bias(fmt);
    if (j.type() != nlohmann::json::value_t::array ||
        j.size() != 2) {
        nnout() << "Invalid bin specification format: " << j.dump() << "\nExiting now.\n";
        exit(1);
    }
    int lower = j[0].get<int>();
    int upper = j[1].get<int>();
    if (lower > upper ||
        lower + bias < 0 ||
        upper + bias >= (1 << get_exp_bits(fmt))) {
        nnout() << "Invalid exponent range [" << lower << "," << upper
                << "] for format " << type_to_string.at(fmt) << "\nExiting now.\n";
        exit(1);
    }
    return BinCounter(lower + bias, upper+bias);
}

void process_bin_spec() {
    using json = nlohmann::json;
    std::ifstream bin_spec_ifs(bin_spec_file);
    json bin_spec_json = json::parse(bin_spec_ifs);
    count_threshold = bin_spec_json["count"].get<unsigned long long int>();

    if (count_threshold <= 0) {
        nnout() << "Invalid count threshold of " << count_threshold << " in bin specification file " << bin_spec_file << "\nExiting now.\n";
        exit(1);
    }

    BinArray host_bins[NUM_FORMATS];
    cudaMalloc(&device_bins, NUM_FORMATS * sizeof(BinArray));
    for (auto fmt : {BF16, FP16, FP32, FP64}) {
        std::string fmt_str = type_to_string.at(fmt);

        std::vector<BinCounter> h_cnts;
        if (bin_spec_json.find(fmt_str) != bin_spec_json.end()) {
            for (const auto& bin_json : bin_spec_json[fmt_str]) {
                h_cnts.push_back(bin_from_json(bin_json, fmt));
            }
        }
        BinCounter* d_cnts;
        size_t to_copy = sizeof(BinCounter) * h_cnts.size();
        cudaMalloc(&d_cnts, to_copy);
        cudaMemcpy(d_cnts, h_cnts.data(), to_copy, cudaMemcpyHostToDevice);
        host_bins[fmt].bins = d_cnts;
        host_bins[fmt].num_bins = h_cnts.size();
    }
    cudaMemcpy(device_bins, host_bins, NUM_FORMATS * sizeof(BinArray), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating device histogram: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

std::string exp_with_bias(char format, int exp);

void tool_init(CUcontext ctx) {
if (histogram_enabled) {
    cudaMalloc(&device_histogram, num_entries * sizeof(unsigned long long int));
    cudaMemset(device_histogram, 0, num_entries * sizeof(unsigned long long int));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating device histogram: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    process_bin_spec();
    channel_host.init(20, sizeof(exp_info), &channel_dev, nullptr);
    recv_thread_running = true;
    recv_thread_receiving = true;
    recv_thread = std::thread(recv_thread_fun<exp_info>,
                              &recv_thread_running,
                              &recv_thread_receiving,
                              channel_host,
                              [](exp_info* data) {
                                  unsigned char fmt = data->format();
                                  std::string fmt_str = type_to_string.at(fmt);
                                  nnout()
                                    << fmt_str << " bin has reached threshold: kernel="
                                    << id_to_kernel[data->kernel_id()]
                                    << " range=[" << exp_with_bias(fmt, data->range().first)
                                    << "," << exp_with_bias(fmt, data->range().second)
                                    << "] count=" << data->get_count() << "\n";
                              });
}
}

void instrument(CUcontext ctx, Instr* instr, const std::string& kname) {
if (histogram_enabled) {
    assert(device_histogram != nullptr && "Histogram not initialized!");
    auto reg_infos = instruction_info::get_reginfo(instr);
    if (reg_infos.size() == 0) {
        return;
    }
    if (kernel_to_id.find(kname) == kernel_to_id.end()) {
        uint32_t new_id = kernel_to_id.size();
        kernel_to_id[kname] = new_id;
        id_to_kernel[new_id] = kname;
    }
    /*nixnan_fp_histogram_counter(int pred, BinArray* bins, unsigned long count,
    unsigned long long int* histogram, ChannelDev* channel_dev, int kerid,
    uint32_t arg_count, ...)*/
    nvbit_insert_call(instr, "nixnan_fp_histogram_counter", IPOINT_AFTER);
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_bins), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(count_threshold), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_histogram), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
    nvbit_add_call_arg_const_val32(instr, tobits32(kernel_to_id[kname]), false); // kerid
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
    nvbit_add_call_arg_const_val64(instr, tobits64(count_threshold), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(device_histogram), false);
    nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
    nvbit_add_call_arg_const_val32(instr, tobits32(kernel_to_id[kname]), false); // kerid
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
    recv_thread_running = false;
    recv_thread.join();
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