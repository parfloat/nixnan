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

#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>

#include "nvbit_reg_rw.h"
#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "../utility/common.h"

__device__ static __forceinline__ uint32_t _FPC_FP32_IS_INF(uint32_t reg_val) {
  uint32_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 24;
  mantissa = reg_val << 9;
  mantissa = mantissa >> 9;
  if (exponent == (uint32_t)(255) && mantissa == (uint32_t)(0)) {
    return Ana_INF;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t _FPC_FP32_IS_NAN(uint32_t reg_val) {
  uint32_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 24;
  mantissa = reg_val << 9;
  mantissa = mantissa >> 9;
  if (exponent == (uint32_t)(255) && mantissa != (uint32_t)(0)) {
    return Ana_NAN;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t
_FPC_FP32_IS_SUBNORMAL(uint32_t reg_val) {
  uint32_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 24;
  mantissa = reg_val << 9;
  mantissa = mantissa >> 9;
  if (exponent == (uint32_t)(0) && mantissa != (uint32_t)(0)) {
    return Ana_SUB;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t _FPC_FP64_IS_NAN(uint64_t reg_val) {
  uint64_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 53;
  mantissa = reg_val << 12;
  mantissa = mantissa >> 12;
  if (exponent == (uint64_t)(2047) && mantissa != (uint64_t)(0)) {
    return Ana_NAN;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t _FPC_FP64_IS_INF(uint64_t reg_val) {
  uint64_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 53;
  mantissa = reg_val << 12;
  mantissa = mantissa >> 12;
  if (exponent == (uint64_t)(2047) && mantissa == (uint64_t)(0)) {
    return Ana_INF;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t
_FPC_FP64_IS_SUBNORMAL(uint64_t reg_val) {
  uint64_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 53;
  mantissa = reg_val << 12;
  mantissa = mantissa >> 12;
  if (exponent == (uint64_t)(0) && mantissa != (uint64_t)(0)) {
    return Ana_SUB;
  }
  return 0;
}

extern "C" __device__ __noinline__ void
fp32_except(int32_t num_regs, int after_before, int pred, int opcode_id,
            int kernel_id, int loc_id, uint64_t pchannel_dev,
            uint32_t with_lit_except...) {
  if (!pred) {
    return;
  }
  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  except_type_info_t ei;
  int4 cta = get_ctaid();
  ei.cta_id_x = cta.x;
  ei.loc_id = loc_id;
  ei.opcode_id = opcode_id;
  ei.kernel_id = kernel_id;
  ei.num_regs = num_regs;
  ei.with_lit_except = with_lit_except;
  ei.after_before = after_before;

  if (num_regs) {
    va_list vl;
    va_start(vl, with_lit_except);
    // printf("num_regs = %d\n", num_regs);
    for (int i = 0; i < num_regs; i++) {
      uint32_t exce = 0;
      uint32_t val = va_arg(vl, uint32_t);
      exce = _FPC_FP32_IS_NAN(val);
      exce = exce + _FPC_FP32_IS_INF(val);
      exce = exce + _FPC_FP32_IS_SUBNORMAL(val);
      for (int tid = 0; tid < 32; tid++) {
        ei.reg_types[tid][i] = __shfl_sync(active_mask, exce, tid);
      }
      // if(exce!=0){
      //     printf("Exception is %u\n",exce);
      // }
    }
    va_end(vl);
  }
  if (first_laneid == laneid) {
    for (int tid = 0; tid < 32; tid++) {
      bool is_transfer = ei.with_lit_except;
      for (int i = 0; i < ei.num_regs; i++) {
        is_transfer = is_transfer || ei.reg_types[tid][i];
      }
      if (is_transfer) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ei, sizeof(except_type_info_t));
        break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
fp64_except(int32_t num_regs, int after_before, int pred, int opcode_id,
            int kernel_id, int loc_id, uint64_t pchannel_dev,
            uint32_t with_lit_except...) {
  if (!pred) {
    return;
  }
  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  except_type_info_t ei;
  int4 cta = get_ctaid();
  ei.cta_id_x = cta.x;
  ei.loc_id = loc_id;
  ei.opcode_id = opcode_id;
  ei.kernel_id = kernel_id;
  ei.num_regs = num_regs;
  ei.with_lit_except = with_lit_except;
  ei.after_before = after_before;
  //  if (num_regs) {
  va_list vl;
  va_start(vl, with_lit_except);
  // printf("num_regs = %d\n", num_regs);
  for (int i = 0; i < num_regs; i++) {
    uint32_t exce = 0;
    uint32_t val_low = va_arg(vl, uint32_t);
    uint32_t val_hi = va_arg(vl, uint32_t);
    uint64_t fp64_val = (uint64_t)val_hi << 32 | val_low;
    exce = _FPC_FP64_IS_NAN(fp64_val);
    exce = exce + _FPC_FP64_IS_INF(fp64_val);
    exce = exce + _FPC_FP64_IS_SUBNORMAL(fp64_val);
    for (int tid = 0; tid < 32; tid++) {
      ei.reg_types[tid][i] = __shfl_sync(active_mask, exce, tid);
    }
    // if(exce!=0){
    //     printf("Exception is %u\n",exce);
    // }
  }
  va_end(vl);
  //  }
  if (first_laneid == laneid) {
    for (int tid = 0; tid < 32; tid++) {
      bool is_transfer = ei.with_lit_except;
      for (int i = 0; i < ei.num_regs; i++) {
        is_transfer = is_transfer || ei.reg_types[tid][i];
      }
      if (is_transfer) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ei, sizeof(except_type_info_t));
        break;
      }
    }
  }
}
