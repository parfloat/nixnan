#ifndef NIXNAN_BARINSTRUMENTATION_H
#define NIXNAN_BARINSTRUMENTATION_H

#include "nvbit.h"
#include "recording.h"
#include "utils/channel.hpp"

bool is_barrier_instruction(Instr* instr);
void instrument_barrier_instruction(Instr* instr, CUcontext ctx, CUfunction func,
                                   std::shared_ptr<nixnan::recorder> recorder, ChannelDev& channel_dev);

#endif // NIXNAN_BARINSTRUMENTATION_H