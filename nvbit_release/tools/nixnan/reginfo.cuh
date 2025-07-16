#ifndef REGINFO_CUH
#define REGINFO_CUH
#include <cstdint>

struct reginfo {
    int count: 6;
    int32_t type: 4;
    bool half_h0: 1;
    bool half_h1: 1;
    bool div0: 1;
    unsigned int operand: 2;
    int num_regs: 4;
private:
    int32_t reserved: 13;
};
static_assert(sizeof(reginfo) == sizeof(int32_t));
#endif // REGINFO_CUH