#ifndef NIXNAN_RECORDING_H
#define NIXNAN_RECORDING_H
#include "common.cuh"
#include "reginfo.cuh"
#include "instruction_info.cuh"
#include <cstdint>
#include <string>
#include <nvbit.h>

namespace nixnan {
    constexpr int SH_AMT = (EXCEBITS + OPBITS);
    class device_recorder {
        public:
            device_recorder(const device_recorder&) = default;
            device_recorder& operator=(const device_recorder&) = default;
            device_recorder(device_recorder&&) = default;
            device_recorder& operator=(device_recorder&&) = default;
        
            __device__ inline
            uint32_t record(uint32_t id, uint32_t exce, uint32_t op) const {
                size_t index = id << SH_AMT | op << EXCEBITS | exce;
                return atomicAdd(device_errors + index, 1);
            }
        
            friend class recorder;
        
        private:
            device_recorder() = default;
            uint32_t* device_errors;

        
            __host__ __device__
            explicit device_recorder(uint32_t* errors) : device_errors(errors) {}
    };
    class recorder {
    public:
        recorder() = delete;
        recorder(const recorder&) = delete;
        recorder& operator=(const recorder&) = delete;
        recorder(recorder&&) = delete;
        recorder& operator=(recorder&&) = delete;
        recorder(size_t sz);
        ~recorder();
        uint32_t mk_entry(Instr *instr, const std::vector<std::pair<reginfo, std::vector<reginsertion>>> &regs,
                          CUcontext ctx, CUfunction f, bool is_mem = false);
        uint32_t mk_entry(std::string& instr, std::string& path, std::string& line, std::string& func,
                          char* optypes, bool is_mem = false);
        void free_device();
        void end();
        device_recorder get_device_recorder();
        std::string& get_inst(uint32_t id);
        std::string& get_path(uint32_t id);
        std::string& get_line(uint32_t id);
        std::string& get_func(uint32_t id);
        uint32_t get_type(uint32_t id, uint32_t op);
        uint32_t get_exce(uint32_t id, uint32_t exce, uint32_t op);
        size_t get_size() const { return size; }
        uint32_t * get_host_errors() const {
            return host_errors;
        }
    private:
        size_t size;
        size_t current_entry = 0;
        struct host_entry {
            std::string instr;
            std::string path;
            std::string line;
            std::string func;
            char opertypes[OPERANDS];
            bool is_mem;
        };
        host_entry* inst_info;
        // Notice this is a pointer to a device array
        uint32_t* device_errors;
        uint32_t* host_errors;
    };
    static_assert(sizeof(device_recorder) == sizeof(uint32_t*),
    "device_recorder size does not match size of pointer which is expected for the injection function.");
}
#endif // NIXNAN_RECORDING_H