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
#include <atomic>

#include "recording.h"
#include "exception_info.cuh"
using nixnan::exception_info;
#include "common.cuh"
#include "instruction_info.cuh"
#include "nnout.hh"
#include "meminstrumentation.cuh"

//GG: -- instrumenting barriers 
#include "barrierinstrumentation.cuh"

uint32_t instr_begin_interval = 0;		// instrn range begin : from INSTR_BEGIN
uint32_t instr_end_interval = UINT32_MAX;	// instrn range end   : from INSTR_END
int verbose = 0;	      			// verbose listing : from TOOL_VERBOSE
int func_details = 0;				// enab fn detail  : from ENABLE_FUN_DETAIL
int print_ill_instr = 0;			// pr. except.inst : from PRINT_ILL_INSTR
int sampling = 0;     				// enable sampling : from SAMPLING
bool instrument_mem = false;			// mem instrum	   : from INSTR_MEM
bool line_info = true;				// getline info	   : from LINE_INFO (may crash)

volatile bool recv_thread_started = false;	// pthread vars
volatile bool recv_thread_receiving = false;	// pthread vars

#define CHANNEL_SIZE sizeof(exception_info)	// chan size
#define TABLE_SIZE (1l << 17)			// 2^17
static __managed__ ChannelDev channel_dev;	// allocate channel_dev in managed mem in static scope

//GG-- for barriers
static __managed__ ChannelDev barChannel_dev;  // new channel for barriers

static ChannelHost channel_host;		// allocate host in static scope
//GG-- for host-side for barriers
static ChannelHost barChannel_host;

std::unordered_set<std::string> kernel_whitelist;  // if in white-list, analyze 
std::unordered_set<std::string> kernel_blacklist;  // if in black-list, do not analyze
std::unordered_map<std::string, int> analyzed_kernels;	// analyzed kernels

// pthread_t recv_thread;
std::thread recv_thread;				// recv thread on host side
std::shared_ptr<nixnan::recorder> recorder = nullptr;	// recorder - shared ptr to the recording infra
std::unordered_set<CUfunction> instrumented_functions;	// set of already intrum cuda fns

bool skip_flag = false;	       				// tbd

void nvbit_at_init() { 					// Basically gets the env flags
  static std::atomic<bool> init_once_flag{false};	// Mark added this?
  if (init_once_flag.exchange(true)) {			// tests if initialized, if so return
    return;
  }
  // Disable warning about using CUDA API calls in nvbit_at_init.
  setenv("ACK_CTX_INIT_LIMITATION", "1", 1);				// ensure a 1
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);			// ensure a 1
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
  GET_VAR_INT(instrument_mem, "INSTR_MEM", 0,
              "Instrument memory instructions for NaN/Inf detection");
  std::string filename;
  GET_VAR_STR(
    filename,
    "LOGFILE",
    "Path to the optional log file. Default is to print to stderr. "
    "This is useful for the case when an instrumented program is "
    "capturing stderr."
  );
  GET_VAR_INT(line_info, "LINE_INFO", 0,
              "Enable debug information for source code locations. This may cause crashes, so set this to 0 if you encounter issues.");
  if (!filename.empty()) {
    set_out_file(filename);
  }
  std::string pad(82, '-');
  nnout() << pad << '\n';
}

//---- instrument a kernel function ----
//
void instrument_function(CUcontext ctx, CUfunction func) {		//GG: how instructions are instrumented. Need to add BAR.SYNC
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions =	// related fns obtained
      nvbit_get_related_functions(ctx, func);
  related_functions.push_back(func);

  for (auto f : related_functions) {
    if (!instrumented_functions.insert(f).second) {	// already seen related function; skip over
      continue;
    }

    std::string kname = cut_kernel_name(nvbit_get_func_name(ctx, func));	// kernel name
    if (verbose) {
      auto old_flags = nnout_stream().flags();
      nnout() << "Inspecting function " << nvbit_get_func_name(ctx, f) <<
                   " at address 0x" << std::hex << nvbit_get_func_addr(ctx, f) << std::endl;
      nnout_stream().flags(old_flags);
    }
    //--- Get each instruction in the kernel ---
    for (auto instr : nvbit_get_instrs(ctx, func)){		
      auto reg_infos = instruction_info::get_reginfo(instr);	// Get their reg info
      bool meminstr = is_memory_instruction(instr);		// see if mem instr
      //GG: Added this
      bool barrier_instr = is_barrier_instruction(instr);	// see if barrier instr
      //if (barrier_instr) nnout() << "BAR found in nixnan"; else nnout() << "BAR not found in nixnan";      
      //---
      if (reg_infos.empty() && !meminstr && !barrier_instr)
      { if (verbose)
        { nnout() << "No instrum.: not mem or bar or has no regs. instr = " << instr << std::endl;
        }
	continue;
      }
      //---
      if (verbose) {
        nnout() << "Instrumenting instruction " << instr->getSass() << std::endl;
      }
      if (meminstr) {						// separate fn for instrum mem instr
        instrument_memory_instruction(instr, ctx, f, recorder, channel_dev);


      } else if (barrier_instr) {
      	instrument_barrier_instruction(instr, ctx, f, recorder, channel_dev);
	
      }
      else
      {  //-- ALL OTHER INSTRUCTIONS get instrumented here --

        uint32_t inst_id = recorder->mk_entry(instr, reg_infos, ctx, f);
        nvbit_insert_call(instr, "nixnan_check_regs", IPOINT_AFTER);	// CHECK REG AFTER EACH INSTR
        nvbit_add_call_arg_guard_pred_val(instr);
        nvbit_add_call_arg_const_val64(instr, tobits64(recorder->get_device_recorder()), false);
        // std::cerr << "#nixnan: Instrumenting instruction with ID " << inst_id << std::endl;
        nvbit_add_call_arg_const_val32(instr, inst_id, false);
        nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
        if (verbose) {
          nnout() << "Instrumenting instruction with " << 1 + std::get<0>(reg_infos[0]).num_regs << " registers" << std::endl;
        }
        {
          auto [ri, rfuns] = reg_infos[0];
          nvbit_add_call_arg_const_val32(instr, 1 + rfuns.size());
          nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
          if (verbose) {
            nnout() << "Instrumenting operand " << ri.operand << ". div0: " << ri.div0 << ", regs: " << ri.num_regs << ", count: " << ri.count << std::endl;
          }
          for (auto& rfun : rfuns) {
            rfun();
          }
        }

        nvbit_insert_call(instr, "nixnan_check_regs", IPOINT_BEFORE);
        nvbit_add_call_arg_guard_pred_val(instr);
        nvbit_add_call_arg_const_val64(instr, tobits64(recorder->get_device_recorder()), false);
        // std::cerr << "#nixnan: Instrumenting instruction with ID " << inst_id << std::endl;
        nvbit_add_call_arg_const_val32(instr, inst_id, false);
        nvbit_add_call_arg_const_val64(instr, tobits64(&channel_dev), false);
        size_t num_regs = 0;
        for (size_t i = 1; i < reg_infos.size(); ++i) {
          auto [ri, rfuns] = reg_infos[i];
          num_regs += ri.num_regs;
        }
        // This is the number of registers that were sent as arguments, plus the
        // number of reg_info functions, minus the first one.
        if (verbose) {
          nnout() << "Instrumenting instruction with " << num_regs + reg_infos.size() - 1 << " registers" << std::endl;
        }
        nvbit_add_call_arg_const_val32(instr, num_regs + reg_infos.size() - 1);
        for (size_t i = 1; i < reg_infos.size(); ++i) {
          auto [ri, rfuns] = reg_infos[i];
          nvbit_add_call_arg_const_val32(instr, tobits32(ri), true);
          if (verbose) {
            nnout() << "Instrumenting operand " << ri.operand << ". div0: " << ri.div0 << ", regs: " << ri.num_regs << std::endl;
          }
          for (auto& rfun : rfuns) {
            rfun();
          }
        }
      }
    }
  }
}

// Kernel to run to flush the rest of the channel
__global__ void flush_channel() {
  // Push negative cta information to the channel to indicate the end of execution
  exception_info ei(int4{-1, -1, -1, -1}, 0, 0, 0, 0);
  // Generates the following warning:
  // /.../device_atomic_functions.hpp(196): Warning: Cannot do atomic on local memory
  // This is inside the nvbit library.
  channel_dev.push(&ei, sizeof(exception_info));
  /* flush channel */
  channel_dev.flush();
}


/*
 * Receiver thread function
 */

// similarly create a recv_thread_fun for recving barrier info wrt its channel
//
void recv_thread_fun(std::shared_ptr<nixnan::recorder> recorder, ChannelHost channel_host) { // recd thread on host does this
  char *recv_buffer = new char[CHANNEL_SIZE];	       // alloc recv_buffer

  while (recv_thread_started) {			       // while recv_thread_started
    uint32_t num_recv_bytes = 0;		       // keep num_recv_bytes going; incr by sizeof(exception_info)

    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {	// while recv_thread_receiving & received some now
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {	// process all recvd
        exception_info *ei = reinterpret_cast<exception_info*>(&recv_buffer[num_processed_bytes]);
        /* when we get this cta_id_x it means the kernel has completed
          */
        if (ei->to_skip()) {				// second time it is being sent - skip
          num_processed_bytes += sizeof(exception_info);
          continue;
        }
        if (ei->warp() == -1) {
          recv_thread_receiving = false;
          break;
        }
        uint32_t id = ei->inst();
        std::string instr = recorder->get_inst(id);
        std::string func = recorder->get_func(id);
        std::string path = recorder->get_path(id);
        std::string line = recorder->get_line(id);
        std::string type;
        if (ei->type() == UNKNOWN) {
          type = type_to_string.at(recorder->get_type(id, ei->operand()));
        }
        else {
          type = type_to_string.at(ei->type());
        }

        uint32_t exce = ei->exception();
        std::vector<std::string> exceptions;
        if (exce & E_NAN) {
          exceptions.push_back("NaN");
        }
        if (exce & E_INF || exce & E_NINF) {
          std::string prefix = (exce & E_INF) ? "" : "-";
          exceptions.push_back(prefix + "infinity");
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
        std::string source_location = path == "" ? "" : " at " + path + ":" + line;
        nnout() << "error [" << errors << "] detected in operand " << ei->operand() << " of instruction " << instr << " in function "
                  << func << source_location << " of type " << type << std::endl;
        num_processed_bytes += sizeof(exception_info);
      }
    }
  }
  delete[] recv_buffer;
  return;
}

/*---------- HOW TO obtain things we need from params ---------------------------------------------
 * See https://claude.ai/share/ee6fb49f-ba11-4cab-9ebe-8c0880f7e48a
 --------------------------------------------------------------------------------------------------
if (tool_has_nvbit_at_cuda_event) {
        cuLaunchKernel_params params;
        params.f = f;
        params.gridDimX = gridDimX;
        params.gridDimY = gridDimY;
        params.gridDimZ = gridDimZ;
        params.blockDimX = blockDimX;
        params.blockDimY = blockDimY;
        params.blockDimZ = blockDimZ;
        params.sharedMemBytes = sharedMemBytes;
        params.hStream = hStream;
        params.kernelParams = kernelParams;
        params.extra = extra;
        
        CUresult status = CUDA_SUCCESS;
        
        nvbit_at_cuda_event(
            ctx,                           // Current CUDA context
            0,                             // is_exit = 0 (entry)
            API_CUDA_cuLaunchKernel,       // Callback ID
            "cuLaunchKernel",              // Function name
            &params,                       // Pointer to parameters
            &status                        // Status (not valid yet)
        );
----------------------------------------------------------------------------------------------------*/
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,	// all events bring here
  const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)								// see when set
    return;

  if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
      cbid == API_CUDA_cuLaunchCooperativeKernelMultiDevice) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;	// gather params for these cases

    if (!is_exit) {	       // cuda event trig NOT at driver API call end -> then only proceed WITH Instrumentation Logic
      /*----- Instrumentation Logic --------- */
      std::string kernel_name = nvbit_get_func_name(ctx, p->f);	// kernel name obtained
      std::string short_name = cut_kernel_name(kernel_name);	// shorten kernel name
      bool enable_instr = false;	// is this a toggle?
      recv_thread_receiving = true;	// is this handshake?

      if (!kernel_whitelist.empty()) {
        enable_instr = kernel_whitelist.count(short_name);	// If user has whitelist and this kernel in it, then this kernel will be instrumented
      } else if (!kernel_blacklist.empty()) {
        enable_instr = !kernel_blacklist.count(short_name);	// If user has a blacklist and kernel in it, then this kernel won't be instrumented
      } else {
        enable_instr = true;					// default true : this kernel will be instrumented
      }

      if (sampling != 0 && analyzed_kernels.count(short_name)) {	// sampling modulo counter
        if (analyzed_kernels[short_name] % sampling != 0) {
          ++analyzed_kernels[short_name];
          enable_instr = false;						// if not turn, not enab
        }
      }

      if (enable_instr) {		// finally if instrumentation is enabled
        instrument_function(ctx, p->f);	// GG: MAIN THING: get the function AND INSTRUMENT IT!
        // Initialize kernel count if not present, then increment
        int count = analyzed_kernels[short_name]++;			// count num kernels instrum
        if (count == 0) {
          nnout() << "running kernel [" << short_name << "] ..." << std::endl;	// Announce first time
        } else if (func_details) {
          nnout() << "running kernel [" << kernel_name << "] ..." // OR based on env var
                    << std::endl;
        }
        ++analyzed_kernels[short_name];			// keep tally of num times instrumented
      }
      nvbit_enable_instrumented(ctx, p->f, enable_instr); // Switch to instrumented if enable_instr
      /*------------ End of Instrumentation Logic ---------------*/
    } else { // ALL executions (instrumented or not) will come here on exit-time event
    
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      cudaError_t kernelError = cudaGetLastError(); // Get error status for last launch
      if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        assert(0);
      }

      /* make sure we prevent re-entry on the nvbit_callback when issuing
      * the flush_channel kernel */
      // skip_flag = true;

      // /* issue flush of channel so we are sure all the memory accesses
      // * have been pushed */
      // flush_channel<<<1, 1>>>();
      // cudaDeviceSynchronize();
      // assert(cudaGetLastError() == cudaSuccess);

      // /* unset the skip flag */
      // skip_flag = false;

      /* wait here until the receiving thread has not finished with the
      * current kernel */
      // while (recv_thread_receiving) {
      //   sched_yield();
      // }
    } // End of DO things relevant to is_exit 
  }
}

/*
 * If already initialized, do not do subsequent steps
 * Else initialize by announcing what's instrumented, etc
 * Mainly : create AND start the receiver thread
 */
 
void nvbit_tool_init(CUcontext ctx) {
  static std::atomic<bool> init_once_flag{false};
  if (init_once_flag.exchange(true)) {
    return;
  }
  std::string k_whitelist_name = "kernel_whitelist.txt";
  std::string k_blacklist_name = "kernel_blacklist.txt";

  nnout() << "Initializing GPU context...\n";
  kernel_whitelist = read_from_file(k_whitelist_name);
  kernel_blacklist = read_from_file(k_blacklist_name);
  if (!kernel_whitelist.empty()) {
  nnout() << "Only instrumenting kernels specified in "
              << k_whitelist_name << std::endl;
  } else if (!kernel_blacklist.empty()) {
  nnout() << "Not instrumenting kernels specified in "
              << k_blacklist_name << std::endl;
  } else {
  nnout() << "Instrumenting all kernels" << std::endl;
  }
  recorder = std::make_shared<nixnan::recorder>(TABLE_SIZE);
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL); // comes from NVBIT; 0 => 1 for barriers
  // GG new channel for barriers below
  barChannel_host.init(1, CHANNEL_SIZE, &barChannel_dev, NULL); // goes with barChannel_host that's allocated above
  // the host channel now knows what dev channel ptr it goes with

  
  recv_thread = std::thread(recv_thread_fun, recorder, channel_host); //GG: Start receive thread!
  // GG: Similarly receive from the bar struct
  // Change CHANNEL_SIZE for barChannel_host... to match the payload size of bar-channel-info-struct-size-as-sizeof
  // i.e. sizeof(bar_exception_info)

  // Join threads into main when done, check status etc..
}

/*
 * When the kernel finishes, run this
 * Basically to read out and print the collected exceptions
 */
void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    recv_thread.join();
  }
  // Similarly join the other recv thread for bar-channel stuff also
  // for now use nnout() << and later somehow serialize
  
  recorder->end();
  recorder->free_device();
  size_t num_inst = recorder->get_size();

  std::map<std::pair<uint32_t, bool>, std::array<std::pair<uint32_t, uint32_t>, EXCEBITS>> exception_counts;
  for (auto ftype : {FP16, BF16, FP32, FP64}) {
    for (bool is_mem : {false, true}) {
      exception_counts[{ftype, is_mem}] = {};
    }
  }
  std::array<std::pair<uint32_t, size_t>, EXCEBITS> exce_idxs = {
    std::make_pair(E_NAN, 0),
    std::make_pair(E_INF, 1),
    std::make_pair(E_NINF, 2),
    std::make_pair(E_SUB, 3),
    std::make_pair(E_DIV0, 4)
  };
  for (size_t i = 0; i < num_inst; ++i) {
    for (int op = 0; op < OPERANDS; ++op) {
      for (int exce = 0; exce < (1<<EXCEBITS); exce++) {
        size_t errors = recorder->get_exce(i, exce, op);
        if (errors == 0) continue;
        uint32_t type = recorder->get_type(i, op);
        bool is_mem = recorder->is_mem(i);
        auto count_exceptions = [&](uint32_t error, size_t index) {
          if (exce & error) {
            std::get<0>(exception_counts[{type, is_mem}][index]) += errors;
            if (errors > 0) {
              std::get<1>(exception_counts[{type, is_mem}][index])++;
            }
          }
        };
        for (auto [error, index] : exce_idxs) {
          count_exceptions(error, index);
        }
      }
    }
  }
  for (auto ftype : {FP16, BF16, FP32, FP64}) {
      for (size_t i = 0; i < 4; ++i) {
        for (auto is_mem : {false, true}) {
          std::get<0>(exception_counts[{ftype, is_mem}][i]) -= std::get<1>(exception_counts[{ftype, is_mem}][i]);
        }
      }
  }
  nnout() << "Finalizing GPU context...\n\n";

  nnout() << "------------ nixnan Report -----------\n\n";
  
  auto print_type_exceptions = [&](const std::string& type_name, uint32_t type_id, bool is_mem) {
    nnout() << "--- " << type_name << (is_mem ? " Memory " : "") << " Operations ---\n";

    auto ecp = exception_counts[{type_id, is_mem}];
    auto old_flags = nnout_stream().flags();
    nnout_stream() << std::dec;
    nnout() << "NaN:           " << std::setw(10) << std::get<1>(ecp[0]) << " (" << std::get<0>(ecp[0]) << " repeats)\n";
    if (!is_mem) {
      nnout() << "Infinity:      " << std::setw(10) << std::get<1>(ecp[1]) << " (" << std::get<0>(ecp[1]) << " repeats)\n";
      nnout() << "-Infinity:     " << std::setw(10) << std::get<1>(ecp[2]) << " (" << std::get<0>(ecp[2]) << " repeats)\n";
      nnout() << "Subnormal:     " << std::setw(10) << std::get<1>(ecp[3]) << " (" << std::get<0>(ecp[3]) << " repeats)\n";
      nnout() << "Division by 0: " << std::setw(10) << std::get<1>(ecp[4]) << " (" << std::get<0>(ecp[4]) << " repeats)\n\n";
    }
    nnout_stream().flags(old_flags);
  };
  for (bool is_mem : {false, true}) {
    print_type_exceptions("FP16", FP16, is_mem);
    print_type_exceptions("BF16", BF16, is_mem);
    print_type_exceptions("FP32", FP32, is_mem);
    print_type_exceptions("FP64", FP64, is_mem);
  }
}