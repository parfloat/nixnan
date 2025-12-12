#include "nvbit.h"
#include "barrierinstrumentation.cuh"
#include "common.cuh"
#include "nnout.hh"
#include "utils/channel.hpp"
#include "instruction_info.cuh"

using namespace InstrType;

bool is_barrier_instruction(Instr* instr) {
  // nnout() << " checking if barrier; Opcode is  " << instr->getOpcode()      <<  std::endl;
  bool barfound = std::string(instr->getOpcode()).find("BAR") != std::string::npos; // looking for BAR.SYNC; might do more robustly later
  if (barfound) nnout() << "BAR found" << std::endl;
  return barfound;
 
} // https://claude.ai/share/ee6fb49f-ba11-4cab-9ebe-8c0880f7e48a has a more robust set of alternatives

// GG: 
// v0: Just print the pertinent info
// v1: Maintain barrier count per thread block
//
void instrument_barrier_instruction(Instr* instr, CUcontext ctx, CUfunction func,
                                   std::shared_ptr<nixnan::recorder> recorder,
                                   ChannelDev& channel_dev) {
   std::string opcode = instr->getOpcode();
   
   nnout() << "Instrumenting barrier instruction: " << opcode << std::endl;

  //-------------------------adding stuff to get threadIdx.x etc
  // Host side - in nvbit_at_init or similar
  nvbit_insert_call(instr, "my_bar_callback", IPOINT_AFTER);  // Changed to IPOINT_AFTER
  //
  // Pass channel_dev pointer and instruction ID
  nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
  nvbit_add_call_arg_const_val32(instr, (uint32_t)instr->getIdx());
  //
  nnout() << " Inserted my_bar_callback for " << opcode << " (IPOINT_AFTER)" << std::endl;	
  // Add launch configuration values as arguments
  /*---
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_TID_X);
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_TID_Y);
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_TID_Z);
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_CTAID_X);
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_CTAID_Y);
  nvbit_add_call_arg_launch_val(instr, NVBIT_CALL_ARG_LAUNCH_CTAID_Z);
  ---*/
 }


/******* USEFUL REF BELOW *********/

/*-->
  // Thread coordinates within block
  // uint32_t tid_x = nvbit_get_tid(ctx, 0);  // 0 = x dimension
  // uint32_t tid_y = nvbit_get_tid(ctx, 1);  // 1 = y dimension
  // uint32_t tid_z = nvbit_get_tid(ctx, 2);  // 2 = z dimension
  // Block coordinates within grid
  // uint32_t ctaid_x = nvbit_get_ctaid(ctx, 0); // 0 = x dimension
  // uint32_t ctaid_y = nvbit_get_ctaid(ctx, 1); // 1 = y dimension
  // uint32_t ctaid_z = nvbit_get_ctaid(ctx, 2); // 2 = z dimension
  // Warp ID (within the block)
  uint32_t warp_id = (threadIdx.x + threadIdx.y * blockDim.x + 
                      threadIdx.z * blockDim.x * blockDim.y) / 32;
  uint32_t flat_tid = threadIdx.x + threadIdx.y * blockDim.x + 
                      threadIdx.z * blockDim.x * blockDim.y;
  uint32_t laneid = flat_tid & 0x1f;  // flat_tid % 32
  // Lane ID (thread within warp, 0-31)
  // uint32_t laneid = __laneid(); // nvbit_get_laneid(ctx);
  std::string opcode = instr->getOpcode();
     nnout() << "Instrumenting barrier instruction: " << opcode 
     //-------------------------------------------------------
     << " at Block coords[" 
     << blockIdx.x << "," << blockIdx.y << "," << blockIdx.z << "]" 
     //-------------------------------------------------------     
     << " at warp,thread[" 
     << warp_id << "," << laneid << "]" 
     //-------------------------------------------------------     
     <<  " at Thread ccords[" 
     << threadIdx.x << "," << threadIdx.y << "," << threadIdx.z << "]" 
     //-------------------------------------------------------     
     <<  std::endl;
  <--*/
