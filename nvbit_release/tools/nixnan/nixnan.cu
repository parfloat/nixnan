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

#include <unordered_set>
#include <memory>
#include <thread>

#include "recording.h"
#include "exception_info.cuh"
using nixnan::exception_info;
#include "common.cuh"
#include "instruction_info.cuh"

uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int func_details = 0;
int print_ill_instr = 0;
int sampling = 0;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

#define CHANNEL_SIZE (1l << 10)
#define TABLE_SIZE (1l << 17)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

std::unordered_set<std::string> kernel_whitelist;
std::unordered_set<std::string> kernel_blacklist;
std::unordered_map<std::string, int> analyzed_kernels;

// pthread_t recv_thread;
std::thread recv_thread;
std::shared_ptr<nixnan::recorder> recorder = nullptr;
std::unordered_set<CUfunction> instrumented_functions;

bool skip_flag = false;

void nvbit_at_init() {
  // Disable warning about using CUDA API calls in nvbit_at_init.
  setenv("ACK_CTX_INIT_LIMITATION", "1", 1);
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(func_details, "ENABLE_FUN_DETAIL", 0,
              "Enable detailed function information for kernel");
  GET_VAR_INT(print_ill_instr, "PRINT_ILL_INSTR", 0,
              "Print the instruction which caused the exception");
  GET_VAR_INT(
      sampling, "SAMPLING", 0,
      "Instrument a repeat kernel every SAMPLING times");
  std::string pad(82, '-');
  std::cerr << pad << '\n';
}


void instrument_function(CUcontext ctx, CUfunction func) {
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);
  related_functions.push_back(func);

  for (auto f : related_functions) {
    if (!instrumented_functions.insert(f).second) {
      continue;
    }

    std::string kname = cut_kernel_name(nvbit_get_func_name(ctx, func));
    if (verbose) {
      std::cerr << "#nixnan: Inspecting function " << nvbit_get_func_name(ctx, f) <<
                   " at address 0x" << std::hex << nvbit_get_func_addr(f) << std::endl;
    }

    for (auto instr : nvbit_get_instrs(ctx, func)){
      auto reginfo = instruction_info::get_reginfo(instr);
      if (reginfo.empty()) { continue; }
      if (verbose) {
        std::cerr << "#nixnan: Instrumenting instruction " << instr->getSass() << std::endl;
      }
      uint32_t reg_num = 0;
      nvbit_insert_call(instr, "nixnan_check_regs", IPOINT_AFTER);
      nvbit_add_call_arg_guard_pred_val(instr);
      nvbit_add_call_arg_const_val64(instr, tobits64(recorder->get_device_recorder()), false);
      uint32_t inst_id = recorder->mk_entry(instr, reginfo[0].first.type, ctx, f);
      // std::cerr << "#nixnan: Instrumenting instruction with ID " << inst_id << std::endl;
      nvbit_add_call_arg_const_val32(instr, inst_id, false);
      nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
      for (auto [ri, rfuns] : reginfo) {
        if (reg_num > 0 && !ri.div0) continue;
        nvbit_add_call_arg_const_val32(instr, 1 + rfuns.size());
        nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
        for (auto& rfun : rfuns) {
          rfun();
        }
        reg_num++;
      }
    }
  }
}

// Kernel to run to flush the rest of the channel
__global__ void flush_channel() {
  // Push negative cta information to the channel to indicate the end of execution
  exception_info ei(int4{-1, -1, -1, -1}, 0, 0, 0);
  // Generates the following warning:
  // /.../device_atomic_functions.hpp(196): Warning: Cannot do atomic on local memory
  // This is inside the nvbit library.
  channel_dev.push(&ei, sizeof(exception_info));
  /* flush channel */
  channel_dev.flush();
}

void recv_thread_fun(std::shared_ptr<nixnan::recorder> recorder, ChannelHost channel_host) {
  char *recv_buffer = new char[CHANNEL_SIZE];

  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;

    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        exception_info *ei = reinterpret_cast<exception_info*>(&recv_buffer[num_processed_bytes]);
        /* when we get this cta_id_x it means the kernel has completed
          */
        if (ei->warp() == -1) {
          recv_thread_receiving = false;
          break;
        }
        uint32_t id = ei->inst();
        std::string instr = recorder->get_inst(id);
        std::string func = recorder->get_func(id);
        std::string path = recorder->get_path(id);
        std::string line = recorder->get_line(id);
        std::string type = type_to_string.at(recorder->get_type(id));

        uint32_t exce = ei->exception();
        std::vector<std::string> exceptions;
        if (exce & E_NAN) {
          exceptions.push_back("NaN");
        }
        if (exce & E_INF) {
          exceptions.push_back("infinity");
        }
        if (exce & E_SUB) {
          exceptions.push_back("subnormal");
        }
        if (exce & E_DIV0) {
          exceptions.push_back("div0");
        }
        std::string errors;
        for (size_t i = 0; i < exceptions.size(); ++i) {
          errors += exceptions[i];
          if (i != exceptions.size() - 1) errors += ",";
        }
        std::cerr << "#nixnan: error [" << errors << "] detected in instruction " << instr << " in function "
                  << func << " at line " << line << " of type " << type << std::endl;
        num_processed_bytes += sizeof(exception_info);
      }
    }
  }
  delete[] recv_buffer;
  return;
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

      if (!kernel_whitelist.empty()) {
        enable_instr = kernel_whitelist.count(short_name);
      } else if (!kernel_blacklist.empty()) {
        enable_instr = !kernel_blacklist.count(short_name);
      } else {
        enable_instr = true;
      }

      if (sampling != 0 && analyzed_kernels.count(short_name)) {
        if (analyzed_kernels[short_name] % sampling != 0) {
          ++analyzed_kernels[short_name];
          enable_instr = false;
        }
      }

      if (enable_instr) {
        instrument_function(ctx, p->f);
        // Initialize kernel count if not present, then increment
        int count = analyzed_kernels[short_name]++;
        if (count == 0) {
          std::cerr << "#nixnan: running kernel [" << short_name << "] ..." << std::endl;
        } else if (func_details) {
          std::cout << "#nixnan: running kernel [" << kernel_name << "] ..."
                    << std::endl;
        }
        ++analyzed_kernels[short_name];
      }
      nvbit_enable_instrumented(ctx, p->f, enable_instr);
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

void nvbit_tool_init(CUcontext ctx) {
  std::string k_whitelist_name = "kernel_whitelist.txt";
  std::string k_blacklist_name = "kernel_blacklist.txt";

  std::cerr << "#nixnan: Initializing GPU context...\n";
  kernel_whitelist = read_from_file(k_whitelist_name);
  kernel_blacklist = read_from_file(k_blacklist_name);
  if (!kernel_whitelist.empty()) {
    std::cerr << "#nixnan: only instrumenting kernels specified in "
              << k_whitelist_name << std::endl;
  } else if (!kernel_blacklist.empty()) {
    std::cerr << "#nixnan: not instrumenting kernels specified in "
              << k_blacklist_name << std::endl;
  } else {
    std::cerr << "#nixnan: instrumenting all kernels" << std::endl;
  }
  recorder = std::make_shared<nixnan::recorder>(TABLE_SIZE);
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  recv_thread = std::thread(recv_thread_fun, recorder, channel_host);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    recv_thread.join();
  }
  recorder->end();
  recorder->free_device();
  size_t num_inst = recorder->get_size();

  uint32_t *host_errors = recorder->get_host_errors();

  std::map<uint32_t,std::array<uint32_t, 4>> exception_counts;
  exception_counts[FP16] = {};
  exception_counts[FP32] = {};
  exception_counts[FP64] = {};

  for (size_t i = 0; i < num_inst; ++i) {
    for (int exce = 0; exce < 16; exce++) {
      size_t idx = i << 4 | exce;
      if (host_errors[idx] == 0) continue;
      uint32_t type = recorder->get_type(i);
      if (exce & E_NAN) {
        exception_counts[type][0] += 1;
      }
      if (exce & E_INF) {
        exception_counts[type][1] += 1;
      }
      if (exce & E_SUB) {
        exception_counts[type][2] += 1;
      }
      if (exce & E_DIV0) {
        exception_counts[type][3] += 1;
      }

    }
  }

  std::cerr << "#nixnan: Finalizing GPU context...\n\n";

  std::cerr << "#nixnan: ------------ nixnan Report -----------\n\n";

  auto print_type_exceptions = [&](const std::string& type_name, uint32_t type_id) {
    std::cerr << "#nixnan: --- " << type_name << " Operations ---\n";
    std::cerr << "#nixnan: NaN: " << exception_counts[type_id][0] << "\n";
    std::cerr << "#nixnan: Infinity: " << exception_counts[type_id][1] << "\n";
    std::cerr << "#nixnan: Subnormal: " << exception_counts[type_id][2] << "\n";
    std::cerr << "#nixnan: Division by 0: " << exception_counts[type_id][3] << "\n\n";
  };

  print_type_exceptions("FP16", FP16);
  print_type_exceptions("FP32", FP32);
  print_type_exceptions("FP64", FP64);
}