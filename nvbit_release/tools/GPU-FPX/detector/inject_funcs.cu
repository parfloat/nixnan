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

__device__ static __forceinline__
uint16_t _HALF_EXPONENT(uint16_t h) {
  return (h >> 10) & 0x1f;
}

__device__ static __forceinline__
uint16_t _HALF_MANTISSA(uint16_t h) {
  return h & 0x3ff;
}

__device__ static __forceinline__
uint32_t _FPC_FP16_IS_INF(uint16_t h) {
  if(_HALF_EXPONENT(h) == 0x1f && _HALF_MANTISSA(h) == 0) {
    return E_INF;
  }
  return 0;
}

__device__ static __forceinline__
uint32_t _FPC_FP16_IS_NAN(uint16_t h) {
  if(_HALF_EXPONENT(h) == 0x1f && _HALF_MANTISSA(h) != 0) {
    return E_NAN;
  }
  return 0;
}

__device__ static __forceinline__
uint32_t _FPC_FP16_IS_SUBNORMAL(uint16_t h) {
  if(_HALF_EXPONENT(h) == 0x0 && _HALF_MANTISSA(h) != 0) {
    return E_SUB;
  }
  return 0;
}

__device__ static __forceinline__
uint32_t _FPC_FP16_IS_0(uint16_t h) {
  if(_HALF_EXPONENT(h) == 0x0 && _HALF_MANTISSA(h) == 0) {
    return E_DIV0;
  }
  return 0;
}

__device__ static __forceinline__
uint32_t _FPC_FP16_CLASSIFY(uint16_t h) {
  return _FPC_FP16_IS_INF(h) + _FPC_FP16_IS_NAN(h) + _FPC_FP16_IS_SUBNORMAL(h);
}

__device__ static __forceinline__ uint32_t _FPC_FP32_IS_INF(uint32_t reg_val) {
  uint32_t exponent, mantissa;
  exponent = reg_val << 1;
  exponent = exponent >> 24;
  mantissa = reg_val << 9;
  mantissa = mantissa >> 9;
  if (exponent == (uint32_t)(255) && mantissa == (uint32_t)(0)) {
    return E_INF;
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
    return E_NAN;
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
    return E_SUB;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t _FPC_FP32_IS_0(uint32_t reg_val) {
  if (_FPC_FP32_IS_INF(reg_val) != 0 || _FPC_FP32_IS_NAN(reg_val) != 0) {
    return E_DIV0;
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
    return E_NAN;
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
    return E_INF;
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
    return E_SUB;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t _FPC_FP64_IS_0(uint64_t reg_val) {
  if (_FPC_FP64_IS_INF(reg_val) != 0 || _FPC_FP64_IS_NAN(reg_val) != 0) {
    return E_DIV0;
  }
  return 0;
}

__device__ static __forceinline__ uint32_t encode_index(uint32_t mem_index,
                                                        uint32_t exce) {
  return mem_index | exce;
}

// __device__
// static
// __forceinline__
// void send_info(int active_mask, int exec, uint32_t mem_index, const int
// laneid, const int first_laneid, reg_info_t &ri){

//     warp_info_t wi;

//     for (int tid = 0; tid < 32; tid++) {
//         //TODO: only shfl to tid=0
//             wi.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
//             wi.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
//             //printf("exce[i] is %d\n",ri.exce_type[tid]);
//     }

//     /* first active lane pushes information on the channel */
//     if (first_laneid == laneid) {
//         for(int i =0; i< 32; i++){
//             if(wi.exce_type[i]>0){
//                 uint32_t e = wi.exce_type[i] -1;
//                 uint32_t final_index = mem_index | e;
//                 return final_index;
//                 uint32_t table_index = encode_index(wi.mem_index_ar[i],
//                 wi.exce_type[i]);
//                 //uint32_t index_info = device_table[table_index];
//                 //printf("table index is %u\n", table_index);
//                 uint32_t index_info = atomicAdd((unsigned
//                 int*)&device_table[table_index], 1); if(index_info == 0) {
//                     //atomicAdd((unsigned int*)&device_table[table_index],
//                     1); ri.warp_exec_info[i] = table_index; ChannelDev
//                     *channel_dev = (ChannelDev *)pchannel_dev;
//                     channel_dev->push(&ri, sizeof(reg_info_t));
//                 }
//                 // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
//                 // channel_dev->push(&ri, sizeof(reg_info_t));
//                 // break;
//             }
//         }
//     }
// }

extern "C" __device__ __noinline__ void
record_mma_val_16_stand(int pred, int opcode_id, int kernel_id,
                        // uint64_t location,
                        // int loc_id,
                        // ushort k_loc_id,
                        // int32_t inst_type,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t low_add,
                        uint32_t high_add) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  // ri.location = (char*)location;
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  // uint32_t *device_table = (uint32_t *)pdevice_table;
  uint32_t exce = 0;

  // Which part is the x and y components
  exce |= _FPC_FP16_CLASSIFY(low_add & 0xFFFF);
  exce |= _FPC_FP16_CLASSIFY((low_add >> 16) & 0xFFFF);
  exce |= _FPC_FP16_CLASSIFY(high_add & 0xFFFF);
  exce |= _FPC_FP16_CLASSIFY((high_add >> 16) & 0xFFFF);
  // printf("exce is %d\n",exce);
  for (int tid = 0; tid < 32; tid++) {
    // TODO: only shfl to tid=0
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
    // printf("exce[i] is %d\n",ri.exce_type[tid]);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        // printf("table index is %u\n", table_index);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_mma_val_32_stand(int pred, int opcode_id, int kernel_id,
                        // uint64_t location,
                        // int loc_id,
                        // ushort k_loc_id,
                        // int32_t inst_type,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t a,
                        uint32_t b, uint32_t c, uint32_t d) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  // ri.location = (char*)location;
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  // uint32_t *device_table = (uint32_t *)pdevice_table;
  uint32_t exce = 0;

  // Which part is the x and y components
  exce |= _FPC_FP32_IS_NAN(a) | _FPC_FP32_IS_INF(a) | _FPC_FP32_IS_SUBNORMAL(a);
  exce |= _FPC_FP32_IS_NAN(b) | _FPC_FP32_IS_INF(b) | _FPC_FP32_IS_SUBNORMAL(b);
  exce |= _FPC_FP32_IS_NAN(c) | _FPC_FP32_IS_INF(c) | _FPC_FP32_IS_SUBNORMAL(c);
  exce |= _FPC_FP32_IS_NAN(d) | _FPC_FP32_IS_INF(d) | _FPC_FP32_IS_SUBNORMAL(d);
  // printf("exce is %d\n",exce);
  for (int tid = 0; tid < 32; tid++) {
    // TODO: only shfl to tid=0
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
    // printf("exce[i] is %d\n",ri.exce_type[tid]);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        // printf("table index is %u\n", table_index);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_mma_val_64_stand(int pred, int opcode_id, int kernel_id,
                        // uint64_t location,
                        // int loc_id,
                        // ushort k_loc_id,
                        // int32_t inst_type,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t low_add,
                        uint32_t high_add) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  // ri.location = (char*)location;
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  // uint32_t *device_table = (uint32_t *)pdevice_table;
  uint32_t exce = 0;

  // Which part is the x and y components
  uint64_t fp64_val = (uint64_t)high_add << 32 | low_add;

  exce = _FPC_FP64_IS_NAN(fp64_val) | _FPC_FP64_IS_INF(fp64_val) | _FPC_FP64_IS_SUBNORMAL(fp64_val);
  // printf("exce is %d\n",exce);
  for (int tid = 0; tid < 32; tid++) {
    // TODO: only shfl to tid=0
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
    // printf("exce[i] is %d\n",ri.exce_type[tid]);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        // printf("table index is %u\n", table_index);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_reg_val_16_stand(int pred, int opcode_id, int kernel_id,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint16_t val) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  ri.mem_index = mem_index;
  uint32_t exce = 0;

  exce = _FPC_FP16_IS_NAN(val) | _FPC_FP16_IS_INF(val) | _FPC_FP16_IS_SUBNORMAL(val);
  for (int tid = 0; tid < 32; tid++) {
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
      }
    }
  }
}

// Vector half instruction
extern "C" __device__ __noinline__ void
record_reg_val_16x2_stand(int pred, int opcode_id, int kernel_id,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t val) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  ri.mem_index = mem_index;
  uint32_t exce = 0;
  uint16_t lower = val & 0xFFFF;
  uint16_t upper = (val >> 16) & 0xFFFF;
  exce = _FPC_FP16_IS_NAN(lower) | _FPC_FP16_IS_INF(lower) | _FPC_FP16_IS_SUBNORMAL(lower);
  exce |= _FPC_FP16_IS_NAN(upper) | _FPC_FP16_IS_INF(upper) | _FPC_FP16_IS_SUBNORMAL(upper);
  for (int tid = 0; tid < 32; tid++) {
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_reg_val_32_stand(int pred, int opcode_id, int kernel_id,
                        // uint64_t location,
                        // int loc_id,
                        // ushort k_loc_id,
                        // int32_t inst_type,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t low_add,
                        uint32_t high_add) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  // ri.location = (char*)location;
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  uint32_t val_low = low_add;
  uint32_t val_hi = high_add;
  // uint32_t *device_table = (uint32_t *)pdevice_table;
  uint32_t exce = 0;

  // for(int tid =0; tid<32; tid++){
  //     ri.reg_vals[tid][0] = __shfl_sync(active_mask, val_low, tid);
  //     ri.reg_vals[tid][1] = __shfl_sync(active_mask, val_hi, tid);
  // }

  exce = _FPC_FP32_IS_NAN(val_low) | _FPC_FP32_IS_INF(val_low) | _FPC_FP32_IS_SUBNORMAL(val_low);
  // printf("exce is %d\n",exce);
  for (int tid = 0; tid < 32; tid++) {
    // TODO: only shfl to tid=0
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
    // printf("exce[i] is %d\n",ri.exce_type[tid]);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        // printf("table index is %u\n", table_index);
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_reg_val_32_div0(int pred, int opcode_id, int kernel_id,
                       // uint64_t location,
                       // int loc_id,
                       // ushort k_loc_id,
                       // int32_t inst_type,
                       uint64_t pdevice_table, uint32_t mem_index,
                       uint64_t pchannel_dev, uint32_t low_add,
                       uint32_t high_add) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  uint32_t val_low = low_add;
  uint32_t val_hi = high_add;
  uint32_t exce = 0;
  // uint32_t *device_table = (uint32_t *)pdevice_table;

  // for(int tid =0; tid<32; tid++){
  //     ri.reg_vals[tid][0] = __shfl_sync(active_mask, val_low, tid);
  //     ri.reg_vals[tid][1] = __shfl_sync(active_mask, val_hi, tid);
  // }

  exce = _FPC_FP32_IS_0(val_low);
  // printf("val_low = %f\n",(float *)val_hi);
  for (int tid = 0; tid < 32; tid++) {
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
  }
  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    // only transfer if there's an excpeionts
    //  int sum = 0;
    // printf("Checking opcode_id = %d\n",opcode_id);
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_reg_val_64_stand(int pred, int opcode_id, int kernel_id,
                        // uint64_t location,
                        // int loc_id,
                        // ushort k_loc_id,
                        // int32_t inst_type,
                        uint64_t pdevice_table, uint32_t mem_index,
                        uint64_t pchannel_dev, uint32_t low_add1,
                        uint32_t high_add1, uint32_t low_add2,
                        uint32_t high_add2) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();

  // ri.location = (char*)location;
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  uint32_t val_low1 = low_add1;
  uint32_t val_hi1 = high_add1;
  uint32_t val_low2 = low_add2;
  uint32_t val_hi2 = high_add2;
  uint32_t exce = 0;
  // uint32_t *device_table = (uint32_t *)pdevice_table;

  // for(int tid =0; tid<32; tid++){
  //     ri.reg_vals[tid][0] = __shfl_sync(active_mask, val_low, tid);
  //     ri.reg_vals[tid][1] = __shfl_sync(active_mask, val_hi, tid);
  // }

  uint64_t fp64_val1 = (uint64_t)val_hi1 << 32 | val_low1;

  exce |= _FPC_FP64_IS_NAN(fp64_val1) | _FPC_FP64_IS_INF(fp64_val1) | _FPC_FP64_IS_SUBNORMAL(fp64_val1);

  uint64_t fp64_val2 = (uint64_t)val_hi2 << 32 | val_low2;

  exce |= _FPC_FP64_IS_NAN(fp64_val2) | _FPC_FP64_IS_INF(fp64_val2) | _FPC_FP64_IS_SUBNORMAL(fp64_val2);
  for (int tid = 0; tid < 32; tid++) {
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
  }

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // atomicAdd((unsigned int*)&device_table[table_index], 1);
        // uint32_t index_info = device_table[table_index];
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          // atomicAdd((unsigned int*)&device_table[table_index], 1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // printf("index info is %u\n", index_info);

        // break;
      }
    }
  }
}

extern "C" __device__ __noinline__ void
record_reg_val_64_div0(int pred, int opcode_id, int kernel_id,
                       // uint64_t location,
                       // int loc_id,
                       // ushort k_loc_id,
                       // int32_t inst_type,
                       uint64_t pdevice_table, uint32_t mem_index,
                       uint64_t pchannel_dev, uint32_t low_add,
                       uint32_t high_add) {

  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_warpid();
  ri.opcode_id = opcode_id;
  ri.kernel_id = kernel_id;
  // ri.loc_id = loc_id;
  // ri.location = (char*)location;
  // ri.inst_type = inst_type;
  ri.mem_index = mem_index;
  uint32_t val_low = low_add;
  uint32_t val_hi = high_add;
  uint32_t exce = 0;
  // uint32_t *device_table = (uint32_t *)pdevice_table;

  // for(int tid =0; tid<32; tid++){
  //     ri.reg_vals[tid][0] = __shfl_sync(active_mask, val_low, tid);
  //     ri.reg_vals[tid][1] = __shfl_sync(active_mask, val_hi, tid);
  // }
  uint64_t fp64_val = (uint64_t)val_hi << 32 | val_low;
  exce = _FPC_FP64_IS_0(fp64_val);
  for (int tid = 0; tid < 32; tid++) {
    ri.exce_type[tid] = __shfl_sync(active_mask, exce, tid);
    ri.mem_index_ar[tid] = __shfl_sync(active_mask, mem_index, tid);
  }
  if (first_laneid == laneid) {
    for (int i = 0; i < 32; i++) {
      if (ri.exce_type[i] > 0) {
        uint32_t table_index =
            encode_index(ri.mem_index_ar[i], ri.exce_type[i]);
        // uint32_t index_info = device_table[table_index];
        uint32_t *device_table = (uint32_t *)pdevice_table;
        uint32_t index_info =
            atomicAdd((unsigned int *)&device_table[table_index], 1);
        if (index_info == 0) {
          //		    atomicAdd((unsigned int*)&device_table[table_index],
          //1);
          ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
          channel_dev->push(&ri, sizeof(reg_info_t));
          break;
        }
        // ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        // channel_dev->push(&ri, sizeof(reg_info_t));
        // break;
      }
    }
  }
}
