/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
// #include "common.h"
#include "../utility/hostutil.h"
#include <algorithm>

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 10)
#define TABLE_SIZE (131071 * 4 * 4)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;

// int verbose = 0;

int func_detailes = 0;
int sampling = 0;

/* opcode to id map and reverse map  */
std::vector<std::string> enable_kernels;
std::vector<std::string> disable_kernels;
std::string enable_kernels_file = "enable_kernels.txt";
std::string disable_kernels_file = "disable_kernels.txt";
uint32_t *device_table;
uint32_t *host_table;

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(func_detailes, "ENABLE_FUN_DETAIL", 0,
              "Enable detailed function information for kernel");
  GET_VAR_INT(print_ill_instr, "PRINT_ILL_INSTR", 0,
              "Print the instruction which cause the exception");
  GET_VAR_INT(
      sampling, "SAMPLING", 0,
      "Instrument a repeat kernel every x (defined by the users) time ");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    std::string k_full_name = nvbit_get_func_name(ctx, func);
    std::string kname = cut_kernel_name(k_full_name);
    int kernel_id = -1;

    if (kernel_id_map.find(kname) == kernel_id_map.end()) {
      kernel_id = kernel_id_map.size();
      kernel_id_map[kname] = kernel_id;
      id_kernel_map[kernel_id] = kname;
    }
    kernel_id = kernel_id_map[kname];

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n",
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      /* Check the type of instruction */
      uint32_t reg_type; 
      bool fp32_inst = false;
      bool mma_inst = false;
      int div_res = is_DIV(instr->getSass());
      bool check_0 = false;
      if (div_res != 0) {
        check_0 = true;
        if (div_res == 1)
          fp32_inst = true;
        else if (div_res == 2) {
          fp32_inst = false;
        }

      } else if (is_FP32_instruction(instr->getSass())) {
        fp32_inst = true;
        reg_type = FP32;
        // printf("SASS of FP32 is %s\n", instr->getSass());
      } else if (is_FP64_instruction(instr->getSass())) {
        fp32_inst = false;
        reg_type = FP64;
      } else if (is_MMA_INSTRUCTION(instr->getSass(), reg_type)) {
        mma_inst = true;
      } else
        continue;
      inst_count++;
      // if (verbose) {
      //   std::cout << instr->getSass() << std::endl;
      // }

      /* Get line info */
      uint32_t offset = instr->getOffset();
      char *file_name = (char *)malloc(sizeof(char) * FILE_NAME_SIZE);
      file_name[0] = '\0';
      char *dir_name = (char *)malloc(sizeof(char) * PATH_NAME_SIZE);
      dir_name[0] = '\0';
      uint32_t line = 0;
      bool ret_line_info =
          nvbit_get_line_info(ctx, f, offset, &file_name, &dir_name, &line);

      // char * loc_str = nullptr;
      std::string file_n(file_name);
      std::string dir_n(dir_name);
      // free(dir_name);
      // free(file_name);

      uint32_t loc_id = 0;

      // ushort k_loc_id = 0;

      if (ret_line_info) {
        LocationTuple loc(file_n, dir_n, line);
        // loc_str = getLocationString(loc);
        loc_id = getLocationID(loc);
        // k_loc_id=getKernelLocationID(loc,K_loc_to_id_map,K_id_to_loc_map);
      } else {
        LocationTuple loc("unknown_path in [" + kname + "]", "", 0);
        // loc_str = getLocationString(loc);
        loc_id = getLocationID(loc);
        // k_loc_id=getKernelLocationID(loc,K_loc_to_id_map,K_id_to_loc_map);
      }
      if (loc_id > 131071) {
        std::cout << "too many FP locations which gpufpx cannot handle."
                  << std::endl;
        exit(-1);
      }
      // if(verbose){
      //   std::cout << "The location is: " <<
      //   locTupleToLoc(id_to_loc_map[loc_id]) << std::endl;
      // }

      std::vector<int> reg_num_list;
      // printf("op_offset is %d\n",op_offset);

      const InstrType::operand_t *op = instr->getOperand(0);

      if (op->type != InstrType::OperandType::REG)
        continue;

      if (mma_inst) {
        if (reg_type == FP16) {
          reg_num_list.push_back(op->u.reg.num);
          reg_num_list.push_back(op->u.reg.num + 1);
        } else if (reg_type == FP32 || reg_type == FP64) {
          reg_num_list.push_back(op->u.reg.num);
          reg_num_list.push_back(op->u.reg.num + 1);
          reg_num_list.push_back(op->u.reg.num + 2);
          reg_num_list.push_back(op->u.reg.num + 3);
        }
      } else if (check_0 && !fp32_inst) {
        reg_num_list.push_back(op->u.reg.num - 1);
        reg_num_list.push_back(op->u.reg.num);
        // printf("fp64 here\n");
      } else {
        reg_num_list.push_back(op->u.reg.num);
        if (!fp32_inst)
          // FP64 requires two registers
          reg_num_list.push_back(op->u.reg.num + 1);
        else
          // We won't use it if it's FP32
          reg_num_list.push_back(op->u.reg.num);
      }

      int opcode_id = 0;
      if (print_ill_instr) {
        if (sass_to_id_map.find(instr->getSass()) == sass_to_id_map.end()) {
          opcode_id = sass_to_id_map.size();
          sass_to_id_map[instr->getSass()] = opcode_id;
          id_to_sass_map[opcode_id] = std::string(instr->getSass());
        }
        opcode_id = sass_to_id_map[instr->getSass()];
      }
      // if(line == 147) {
      // std::cout << "instruction in line 147 is " << instr->getSass() << ",
      // opcode is "<< opcode_id << std::endl;
      // }
      uint32_t index = encode_index(loc_id, reg_type);
      // std::cout << instr->getSass() << ", " << reg_type << '\n';
      /* insert call to the instrumentation function with its
       * arguments */
      if (mma_inst) {
        if (reg_type == FP16) {
          nvbit_insert_call(instr, "record_mma_val_16_stand", IPOINT_AFTER);
        } else if (reg_type == FP32) {
          nvbit_insert_call(instr, "record_mma_val_32_stand", IPOINT_AFTER);
        } else if (reg_type == FP64) {
          nvbit_insert_call(instr, "record_reg_val_64_stand", IPOINT_AFTER);
        } else {
          std::cout << "Unknown MMA register type\n";
          exit(1);
        }
      } else if (fp32_inst && !check_0) {
        nvbit_insert_call(instr, "record_reg_val_32_stand", IPOINT_AFTER);
      } else if (fp32_inst && check_0) {
        nvbit_insert_call(instr, "record_reg_val_32_div0", IPOINT_AFTER);
      } else if (!fp32_inst && !check_0) {
        nvbit_insert_call(instr, "record_reg_val_64_stand", IPOINT_AFTER);
      } else if (!fp32_inst && check_0) {
        nvbit_insert_call(instr, "record_reg_val_64_div0", IPOINT_AFTER);
      } else {
        std::cout << "Unsupported checking!" << std::endl;
        exit(-1);
      }

      /* guard predicate value */
      nvbit_add_call_arg_guard_pred_val(instr);
      /* opcode id */
      nvbit_add_call_arg_const_val32(instr, opcode_id);
      nvbit_add_call_arg_const_val32(instr, kernel_id);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)device_table);
      nvbit_add_call_arg_const_val32(instr, index);
      /* add pointer to channel_dev*/
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);

      for (size_t i = 0; i < reg_num_list.size(); i++) {
        nvbit_add_call_arg_reg_val(instr, reg_num_list[i], false);
      }
      // cnt++;
    }
  }
}

__global__ void flush_channel() {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  reg_info_t ri;
  ri.cta_id_x = -1;
  channel_dev.push(&ri, sizeof(reg_info_t));

  /* flush channel */
  channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
      cbid == API_CUDA_cuLaunchCooperativeKernelMultiDevice) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    if (!is_exit) {
      /*----- Instrumentation Logic --------- */
      std::string kernel_name = nvbit_get_func_name(ctx, p->f);
      std::string short_name = cut_kernel_name(kernel_name);
      bool enable_instr = false;
      recv_thread_receiving = true;

      if (!enable_kernels.empty()) {
        if (std::find(enable_kernels.begin(), enable_kernels.end(),
                      short_name) != enable_kernels.end()) {
          enable_instr = true;
        }
      } else if (!disable_kernels.empty()) {
        if (std::find(disable_kernels.begin(), disable_kernels.end(),
                      short_name) == disable_kernels.end()) {
          enable_instr = true;
        }
      } else {
        enable_instr = true;
      }

      // recv_thread_receiving = true;
      if (sampling) {
        // printf("Instrument every %d repeat for a kernel", sampling);
        if (analyzed_kernels.find(short_name) != analyzed_kernels.end()) {
          if ((analyzed_kernels[short_name] - 1) % sampling != 0) {
            analyzed_kernels[short_name] = analyzed_kernels[short_name] + 1;
            enable_instr = false;
          }
        }
      }

      if (enable_instr) {
        instrument_function_if_needed(ctx, p->f);
        nvbit_enable_instrumented(ctx, p->f, true);
        if (analyzed_kernels.find(short_name) == analyzed_kernels.end()) {
          analyzed_kernels[short_name] = 1;
          std::cout << "Running #GPU-FPX: kernel [" << short_name << "] ..."
                    << std::endl;
        } else {
          if (func_detailes) {
            std::cout << "Running #GPU-FPX: kernel [" << kernel_name << "] ..."
                      << std::endl;
            // analyzed_kernels.insert(kernel_name).second;
          }
          analyzed_kernels[short_name] = analyzed_kernels[short_name] + 1;
        }
      } else {
        nvbit_enable_instrumented(ctx, p->f, false);
      }

      /*------------ End of Instrumentation Logic ---------------*/

    } else {
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      cudaError_t kernelError = cudaGetLastError();
      if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        assert(0);
      }

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>();
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* unset the skip flag */
      skip_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (recv_thread_receiving) {
        sched_yield();
      }
    }
  }
}

void *recv_thread_fun(void *) {
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;

    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        reg_info_t *ri = (reg_info_t *)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed
         */
        if (ri->cta_id_x == -1) {
          recv_thread_receiving = false;
          break;
        }

        // char *loc = ri->location;
        uint32_t index = ri->mem_index;
        uint32_t loc_id = 0;
        uint32_t inst_type = 0;
        decode_index(index, loc_id, inst_type);
        // int loc_id = ri->loc_id;
        std::string loc = locTupleToLoc(id_to_loc_map[loc_id]);
        std::string fp_type = FPFormatTypeNames[inst_type];
        for (int i = 0; i < 32; i++) {
          if (ri->exce_type[i] == 0)
            continue;
          for (auto exception : mapExceptions(ri->exce_type[i])) {
            print_exc(loc, fp_type, exception.first,
                      ri->opcode_id, ri->kernel_id, inst_type, loc_id,
                      exception.second);
          }
        }

        num_processed_bytes += sizeof(reg_info_t);
      }
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  printf("#GPU-FPX: Initializing GPU context...\n");
  read_from_file(enable_kernels_file, enable_kernels);
  read_from_file(disable_kernels_file, disable_kernels);
  if (!enable_kernels.empty()) {
    std::cout << "#GPU-FPX: Will only instrument the kernels you specify in "
              << enable_kernels_file << std::endl;
  } else if (!disable_kernels.empty()) {
    std::cout << "#GPU-FPX: Won't instrument the kernels you specify in "
              << disable_kernels_file << std::endl;
  } else {
    std::cout << "#GPU-FPX: Instrument all kernels." << std::endl;
  }
  CUDA_SAFECALL(cudaMalloc(&device_table, sizeof(uint32_t) * TABLE_SIZE));
  host_table = (uint32_t *)malloc(sizeof(uint32_t) * TABLE_SIZE);
  for (int i = 0; i < TABLE_SIZE; ++i) {
    host_table[i] = (uint32_t)0;
  }
  CUDA_SAFECALL(cudaMemcpy(device_table, host_table,
                           sizeof(uint32_t) * TABLE_SIZE,
                           cudaMemcpyHostToDevice));
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }

  printf("#GPU-FPX: Finalizing GPU context...\n\n");

  printf("------------ GPU-FPX Report -----------\n\n");

  printf("--- FP16 Operations ---\n");
  printf("Total NaN found:              %d\n", computeExcNum(E_NAN, FP16));
  printf("Total INF found:              %d\n", computeExcNum(E_INF, FP16));
  printf("Total underflow (subnormal):  %d\n", computeExcNum(E_SUB, FP16));
  printf("Total Division by 0:          %d\n", computeExcNum(E_DIV0, FP16));

  printf("--- FP32 Operations ---\n");
  printf("Total NaN found:              %d\n", computeExcNum(E_NAN, FP32));
  printf("Total INF found:              %d\n", computeExcNum(E_INF, FP32));
  printf("Total underflow (subnormal):  %d\n", computeExcNum(E_SUB, FP32));
  printf("Total Division by 0:          %d\n", computeExcNum(E_DIV0, FP32));

  printf("--- FP64 Operations ---\n");
  printf("Total NaN found:              %d\n", computeExcNum(E_NAN, FP64));
  printf("Total INF found:              %d\n", computeExcNum(E_INF, FP64));
  printf("Total underflow (subnormal):  %d\n", computeExcNum(E_SUB, FP64));
  printf("Total Division by 0:          %d\n", computeExcNum(E_DIV0, FP64));

  printf("--- Other Stats ---\n");
  printf("Kernels:      %lu\n", analyzed_kernels.size());
  print_real_exceptions();
  // printf("FP Inst. count:  %d\n", inst_count);
  CUDA_SAFECALL(cudaFree(device_table));
  free(host_table);
}
