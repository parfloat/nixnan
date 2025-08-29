#ifndef NIXNAN_MEMINSTRUMENTATION_H
#define NIXNAN_MEMINSTRUMENTATION_H

#include "nvbit.h"
#include "recording.h"

bool is_memory_instruction(Instr* instr);
void instrument_memory_instruction(Instr* instr, CUcontext ctx, CUfunction func,
                                   std::shared_ptr<nixnan::recorder> recorder, ChannelDev& channel_dev);

#endif // NIXNAN_MEMINSTRUMENTATION_H