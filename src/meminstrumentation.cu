#include "nvbit.h"
#include "meminstrumentation.cuh"
#include "common.cuh"
#include "nnout.hh"
#include "utils/channel.hpp"
#include "instruction_info.cuh"

using namespace InstrType;

bool is_memory_instruction(Instr* instr) {
  return std::string(instr->getOpcode()).find("STG") != std::string::npos;
}

uint32_t find_type(Instr* instr, CUcontext ctx, CUfunction func) {
    auto cfg = nvbit_get_CFG(ctx, func);
    for (auto bb : cfg.bbs) {
        int reg = -1;
        for (auto it = bb->instrs.rbegin(); it != bb->instrs.rend(); ++it) {
            auto inst = *it;
            if (inst == instr) {
                reg = instr->getOperand(1)->u.reg.num;
            } else if (reg != -1) {
                for (auto i = 0; i < inst->getNumOperands(); i++) {
                    auto op = inst->getOperand(i);
                    if (op->type == OperandType::REG && op->u.reg.num == reg) {
                        // We found the prior instruction that used this register.
                        // Speculate that this is the type
                        auto ris = instruction_info::get_reginfo(inst);
                        if (ris.size() == 0) {
                            // Not a floating point instruction.
                            return UNKNOWN;
                        }
                        return ris[i].first.type;
                    }
                }
            }
        }
    }
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
        if (verbose) {
            nnout() << "Unsupported store instruction: " << opcode << std::endl;
        }
        return;
    }
    if (verbose) {
        nnout() << "Instrumenting memory instruction: " << opcode  << " of width " << width << std::endl;
    }
    // Determine possible type information
    uint32_t type = find_type(instr, ctx, func);
    if (type == UNKNOWN) {
        if (verbose) {
            nnout() << "Skipping memory instruction of unknown type: " << opcode << std::endl;
        }
        return;
    } else if (verbose) {
        nnout() << "Found memory instruction of type: " << type_to_string.at(type) << std::endl;
    }
    reginfo ri;
    ri.type = type;
    // Register with the recorder
    std::vector<std::pair<reginfo, std::vector<reginsertion>>> v =
        {std::pair<reginfo, std::vector<reginsertion>>(ri, std::vector<reginsertion>{}),
         std::pair<reginfo, std::vector<reginsertion>>(ri, std::vector<reginsertion>{})};
    uint32_t inst_id = recorder->mk_entry(instr, v, ctx, func, true);

    nvbit_insert_call(instr, "nixnan_check_nans", IPOINT_BEFORE);
    // void nixnan_check_nans(int pred, device_recorder recorder, uint32_t inst_id,
    //                        ChannelDev* pchannel_dev, uint32_t type, uint32_t arg_count, ...)
    nvbit_add_call_arg_guard_pred_val(instr, false);
    nvbit_add_call_arg_const_val64(instr, tobits64(recorder->get_device_recorder()), false);
    nvbit_add_call_arg_const_val32(instr, inst_id, false);
    nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
    nvbit_add_call_arg_const_val32(instr, type, false);
    nvbit_add_call_arg_const_val32(instr, width/32, false);

    int reg = instr->getOperand(1)->u.reg.num;
    // Add all the registers
    for (size_t i = 0; i < width/32; i++) {
        nvbit_add_call_arg_reg_val(instr, reg + i , true);
    }
}