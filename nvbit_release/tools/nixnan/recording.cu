#include "recording.h"
#include "utils/utils.h"
#include <cuda.h>
#include <stdexcept>
#include "common.cuh"

namespace nixnan {
    const uint32_t NUM_EXCE_BITS = 4;
    recorder::recorder(size_t sz) : size(sz), current_entry(0) {
        if (sz > (1 << (32-NUM_EXCE_BITS))) {
            throw std::runtime_error("Requested log size is too large");
        }
        inst_info = new host_entry[sz];
        CUDA_SAFECALL(cudaMalloc((void**)&device_errors, sz * sizeof(uint32_t) << NUM_EXCE_BITS));
        host_errors = new uint32_t[sz << NUM_EXCE_BITS];
        for(size_t i = 0; i < sz << NUM_EXCE_BITS; i++) {
            host_errors[i] = 0;
        }
        CUDA_SAFECALL(cudaMemcpy(device_errors, host_errors, sz * sizeof(uint32_t) << NUM_EXCE_BITS, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }
    recorder::~recorder() {
        delete[] inst_info;
        if (host_errors) {
            delete[] host_errors;
        }
        if (device_errors) {
            CUDA_SAFECALL(cudaFree(device_errors));
            cudaDeviceSynchronize();
        }
    }
    uint32_t recorder::mk_entry(Instr *instr, uint32_t type, CUcontext ctx, CUfunction f) {
        std::string instr_str = instr->getSass();
        uint32_t offset = instr->getOffset();
        char *file_name = (char *)malloc(sizeof(char) * 1024);
        file_name[0] = '\0';
        char *dir_name = (char *)malloc(sizeof(char) * 1024);
        dir_name[0] = '\0';
        uint32_t line = 0;
        bool ret_line_info = nvbit_get_line_info(ctx, f, offset, &file_name, &dir_name, &line);
        std::string path = file_name;
        path += dir_name;
        std::string line_str = std::to_string(line);
        std::string func = cut_kernel_name(nvbit_get_func_name(ctx, f));
        return mk_entry(instr_str, path, line_str, func, type);
    }
    uint32_t recorder::mk_entry(std::string& instr, std::string& path, std::string& line, std::string& func, uint32_t type) {
        inst_info[current_entry].instr = instr;
        inst_info[current_entry].path = path;
        inst_info[current_entry].line = line;
        inst_info[current_entry].func = func;
        inst_info[current_entry].type = type;
        return current_entry++;
    }
    void recorder::free_device() {
        CUDA_SAFECALL(cudaFree(device_errors));
        device_errors = nullptr;
    }
    void recorder::end() {
        CUDA_SAFECALL(cudaMemcpy(host_errors, device_errors, size * sizeof(uint32_t) << NUM_EXCE_BITS, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    device_recorder recorder::get_device_recorder() {
        return device_recorder(device_errors);
    }
    std::string& recorder::get_inst(uint32_t id) {
        return inst_info[id].instr;
    }
    std::string& recorder::get_path(uint32_t id) {
        return inst_info[id].path;
    }
    std::string& recorder::get_line(uint32_t id) {
        return inst_info[id].line;
    }
    std::string& recorder::get_func(uint32_t id) {
        return inst_info[id].func;
    }
    uint32_t recorder::get_type(uint32_t id) {
        return inst_info[id].type;
    }
    uint32_t recorder::get_exce(uint32_t id) {
        return 0;
    }
}