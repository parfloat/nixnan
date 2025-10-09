#include "instruction_info.cuh"
#include <fstream>
#include "common.cuh"
#include <iostream>
#include "instructions.h"
#include <string>
#include <functional>
#include <regex>
#include <cmath>
#include "nnout.hh"

using InstrType::OperandType;

size_t get_num_regs(size_t type, size_t count) {
    switch (type) {
        case FP16:
        case BF16:
            return (count + 1) / 2;
        case FP32:
            return count;
        case FP64:
            return 2 * count;
        default:
            throw std::runtime_error("Unknown bit size for register extraction");
    }
}

size_t clamp(size_t val) {
    return (val > 255) ? 255 : val;
}

std::vector<reginsertion> get_regs(Instr *instr, size_t operand, size_t type, size_t count, reginfo &reg_info, bool f64high = false) {
    std::vector<reginsertion> reg_ops;
    auto num_operands = instr->getNumOperands();
    assert(num_operands > 0);
    assert(operand < num_operands);

    if (operand >= (size_t) num_operands) {
        throw std::runtime_error("Invalid operand index");
    }
    const auto& op = instr->getOperand(operand);
    size_t num_regs = get_num_regs(type, count);
    switch (op->type) {
        case OperandType::REG:
        case OperandType::UREG: {
            size_t reg_start = op->u.reg.num - (f64high ? 1 : 0);
            OperandType type = op->type;
            for (size_t i = 0; i < num_regs; i++) {
                reg_ops.push_back([instr, reg_start, i, type]() {
                    auto fn = type == OperandType::REG ? nvbit_add_call_arg_reg_val : nvbit_add_call_arg_ureg_val;
                    fn(instr, clamp(reg_start+i), true);
                });
            }
            break;
        } case OperandType::IMM_DOUBLE: {
            double val = op->u.imm_double.value;
            uint64_t tmp;
            memcpy(&tmp, &val, sizeof(double));
            reg_ops.push_back([instr, tmp]() {
                nvbit_add_call_arg_const_val32(instr, tmp & 0xFFFFFFFF, true);
                nvbit_add_call_arg_const_val32(instr, (tmp >> 32) & 0xFFFFFFFF, true);
            });
            if (reg_info.type == FP32) {
                num_regs++;
            }
            reg_info.type = FP64;
            break;
        } case OperandType::IMM_UINT64: {
            // This shouldn't be an error???
            num_regs = 0;
            break;
        } case OperandType::CBANK: {
            auto cbank = op->u.cbank;
            for (size_t i = 0; i < num_regs; i++) {
                reg_ops.push_back([instr, cbank, i, f64high]() {
                    int offset = cbank.imm_offset + 4 * (i + (f64high ? -1 : 0));
                    nvbit_add_call_arg_cbank_val(instr, cbank.id, offset, true);
                });
            }
            break;
        } case OperandType::GENERIC: {
            auto x = op->u.generic.array;
            std::string x_str(x);
            if (x_str.find("NAN") != std::string::npos) {
                // This is a NAN register, we don't need to do anything
                num_regs = 0;
                std::cerr << "#nixnan: NaN immediate found in operand " << op->str << std::endl;
            } else if (x_str.find("INF") != std::string::npos) {
                // This is an INF register, we don't need to do anything
                num_regs = 0;
                std::cerr << "#nixnan: Infinite immediate found in operand " << op->str << std::endl;
            }
            break;
        }
        default: {
            nnout() << "Unsupported operand type for register extraction: " << (int)op->type << std::endl;
            break;
        }
    }
    reg_info.num_regs = num_regs;
    return reg_ops;
}

bool is_f64high(std::string opcode_str) {
    std::string f64high_suffix = "64H";
    size_t num_chars = f64high_suffix.size();
    return opcode_str.size() >= num_chars && opcode_str.rfind(f64high_suffix) == opcode_str.size() - num_chars;
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
            ri.operand = i;
            // ri.num_regs = get_num_regs(ri.type, ri.count);
            assert(get_num_regs(ri.type, ri.count) < 16);
            std::string opcode_str(instr->getOpcode());
            ri.f64high = is_f64high(opcode_str);
            auto reg_info = get_regs(instr, i, ri.type, ri.count, ri, ri.f64high);
            if (ri.num_regs > 0) {
                reg_infos.push_back({ri, reg_info});
            }
        }
    }
    return reg_infos;
}