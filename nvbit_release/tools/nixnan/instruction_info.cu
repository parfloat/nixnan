#include "instruction_info.cuh"
#include <fstream>
#include "common.cuh"
#include <iostream>
#include "instructions.h"
#include <string>
#include <functional>
#include <regex>

using InstrType::OperandType;

std::vector<reginsertion> get_regs(Instr *instr, size_t operand, size_t type, size_t count) {
    std::vector<reginsertion> reg_ops;
    auto num_operands = instr->getNumOperands();
    assert(num_operands > 0);

    if (operand >= (size_t) num_operands) {
        throw std::runtime_error("Invalid operand index");
    }
    const auto& op = instr->getOperand(operand);
    size_t num_regs = 0;
    if (type == FP16 || type == BF16) {
        num_regs = (count + 1) / 2;
    } else if (type == FP32) {
        num_regs = count;
    } else if (type == FP64) {
        num_regs = 2*count;
    } else {
        throw std::runtime_error("Unknown bit size for register extraction");
    }
    switch (op->type) {
        case OperandType::REG: {
            size_t reg_start = op->u.reg.num;
            for (size_t i = 0; i < num_regs; i++) {
                reg_ops.push_back([instr, reg_start, i]() {
                    nvbit_add_call_arg_reg_val(instr, reg_start+i, true);
                });
            }
            break;
        } case OperandType::IMM_DOUBLE: {
            double val = op->u.imm_double.value;
            uint64_t tmp;
            memcpy(&tmp, &val, sizeof(double));
            reg_ops.push_back([instr, tmp]() {
                nvbit_add_call_arg_const_val32(instr, (tmp >> 32) & 0xFFFFFFFF);
                nvbit_add_call_arg_const_val32(instr, tmp & 0xFFFFFFFF);
            });
            break;
        } case OperandType::IMM_UINT64: {
            // This shouldn't be an error???
            break;
        } case OperandType::CBANK: {
            size_t bank_start = op->u.cbank.id;
            for (size_t i = 0; i < num_regs; i++) {
                reg_ops.push_back([instr, bank_start, i]() {
                    nvbit_add_call_arg_reg_val(instr, bank_start+i, true);
                });
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported operand type for register extraction: " + std::to_string(static_cast<int>(op->type)));
    }
    return reg_ops;
}


std::vector<std::pair<reginfo, std::vector<reginsertion>>> instruction_info::get_reginfo(Instr *instr) {
    static nlohmann::json instructions;
    if (instructions.empty()) {
        std::stringstream(std::string(instructions_json,
            instructions_json + instructions_json_len)) >> instructions;
    }
    std::vector<std::pair<reginfo, std::vector<reginsertion>>> reg_infos;
    auto inst_name = std::string(instr->getOpcode());
    
    std::string matched_key;
    for (auto it = instructions.begin(); it != instructions.end(); ++it) {
        std::regex re(it.key());
        if (std::regex_match(inst_name, re)) {
            matched_key = it.key();
            break;
        }
    }
    if (!matched_key.empty()) {
        inst_name = matched_key;
    }

    if (instructions.contains(inst_name)) {
        const auto& instruction_data = instructions[inst_name];
        for (size_t i = 0; i < instruction_data["registers"].size(); i++) {
            const auto& reg = instruction_data["registers"][i];
            reginfo ri;
            ri.half_h0 = true;
            ri.half_h1 = true;
            ri.count = reg["count"].get<int>();
            ri.type = string_to_type.at(reg["type"].get<std::string>());
            ri.div0 = reg.find("div0") != reg.end();
            reg_infos.push_back({ri, get_regs(instr, i, ri.type, ri.count)});
        }
    }
    return reg_infos;
}