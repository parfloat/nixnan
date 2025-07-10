#ifndef REGINFO_CUH
#define REGINFO_CUH
#include <cstdint>

struct reginfo {
    int count: 6;
    int32_t type: 4;
    bool half_h0: 1;
    bool half_h1: 1;
    bool div0: 1;
    int operand: 2;
private:
    int32_t reserved: 17;
};
static_assert(sizeof(reginfo) == sizeof(int32_t));
#endif // REGINFO_CUH