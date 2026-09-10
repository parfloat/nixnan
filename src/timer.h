#ifndef TIMER_H
#define TIMER_H

#include "nvbit.h"

namespace nixnan{
namespace timer {
    void init();
    void tool_init(CUcontext ctx);
    void instrument(CUcontext ctx, Instr* instr, const std::string& kname);
    void term();
}
}

#endif // TIMER_H