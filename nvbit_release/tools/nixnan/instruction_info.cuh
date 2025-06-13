#ifndef INSTRUCTION_INFO_H
#define INSTRUCTION_INFO_H

#include <vector>
#include <string>
#include "reginfo.cuh"
#include "nlohmann/json.hpp"
#include "nvbit.h"

typedef std::function<void()> reginsertion;

class instruction_info {
private:
    instruction_info() = delete;
    instruction_info(const instruction_info&) = delete;
    instruction_info& operator=(const instruction_info&) = delete;
    instruction_info(instruction_info&&) = delete;
    instruction_info& operator=(instruction_info&&) = delete;

public:
    static std::vector<std::pair<reginfo, std::vector<reginsertion>>> get_reginfo(Instr *instr);
};

#endif // INSTRUCTION_INFO_H