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
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  int32_t kernel_id;
  // int32_t num_regs;
  /* 32 lanes, each thread can store up to 5 register values */
  // uint32_t reg_vals[32][8];
  /* code location (file, line) */
  // char *location;
  // int32_t loc_id;
  /* Instruction type: FP32=1, FP64=2 */
  // int32_t inst_type;
  uint32_t mem_index;
  uint32_t mem_index_ar[32];
  uint32_t exce_type[32];
  uint32_t warp_exec_info[32];
  // uint32_t reg_vals[32][2];
} reg_info_t;

typedef struct {
  uint32_t mem_index_ar[32];
  uint32_t exce_type[32];
  // uint32_t reg_vals[32][2];
} warp_info_t;

typedef struct {
  int32_t cta_id_x;
  int32_t opcode_id;
  int32_t kernel_id;
  int32_t loc_id;
  int32_t num_regs;
  uint32_t with_lit_except;
  uint32_t after_before;
  uint32_t reg_types[32][4];
} except_type_info_t;

const uint32_t E_NAN = 1,
  E_INF = 2,
  E_SUB = 4,
  E_DIV0 = 8;
enum ExceptionAnaType {
  Ana_SUB =
      0, /*Consider SUB as a normal val now. Will support SUB in the future*/
  Ana_NAN,
  Ana_INF,
  NUM_ANA_TYPES,
};

const uint32_t FP16 = 0,
  FP32 = 1,
  FP64 = 2;

#endif
