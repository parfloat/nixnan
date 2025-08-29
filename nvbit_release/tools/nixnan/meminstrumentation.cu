#include "nvbit.h"
#include "meminstrumentation.cuh"
#include "common.cuh"
#include "nnout.hh"
#include "utils/channel.hpp"

bool is_memory_instruction(Instr* instr) {
  return std::string(instr->getOpcode()).find("STG") != std::string::npos;
}

uint32_t find_type(Instr* instr, CUcontext ctx, CUfunction func) {
    nvbit_get_CFG(ctx, func);
    return UNKNOWN;
}

void instrument_memory_instruction(Instr* instr, CUcontext ctx, CUfunction func,
                                   std::shared_ptr<nixnan::recorder> recorder,
                                   ChannelDev& channel_dev) {
    size_t width = 32;
    std::string opcode = instr->getOpcode();
    if (opcode.find("128") != std::string::npos) {
        width = 128;
    } else if (opcode.find("64") != std::string::npos) {
        width = 64;
    } else if (opcode.find("U8") != std::string::npos ||
               opcode.find("U16") != std::string::npos) {
        nnout() << "Unsupported store instruction: " << opcode << std::endl;
        return;
    }

    // Determine possible type information
    uint32_t type = find_type(instr, ctx, func);

    // Register with the recorder
    std::vector<std::pair<reginfo, std::vector<reginsertion>>> v;
    recorder->mk_entry(instr, v, ctx, func);
    uint32_t inst_id = 0;

    nvbit_insert_call(instr, "nixnan_check_nans", IPOINT_BEFORE);
    // void nixnan_check_nans(int pred, device_recorder recorder, uint32_t inst_id,
    //                        ChannelDev* pchannel_dev, uint32_t type, uint32_t arg_count, ...)
    nvbit_add_call_arg_guard_pred_val(instr, false);
    nvbit_add_call_arg_const_val64(instr, tobits64(recorder->get_device_recorder()), false);
    nvbit_add_call_arg_const_val32(instr, inst_id, false);
    nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
    nvbit_add_call_arg_const_val32(instr, type, false);
    nvbit_add_call_arg_const_val32(instr, width/32, false);
    // Add all the registers
    for (size_t i = 0; i < width/32; i++) {
        nvbit_add_call_arg_reg_val(instr, i, true);
    }
}