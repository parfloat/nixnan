/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * cuFuzz NVBit Injection Functions
 * Device-side instrumentation for edge coverage collection.
 *
 * Author: Mohamed Tarek (mtarek@nvidia.com)
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

#define MAP_SIZE 65536
#define MAX_BBS MAP_SIZE

extern "C" __device__ __noinline__ void record_coverage_edge_count(int bb_id,
                                                                   uint64_t p_exec_cov_bb, 
                                                                   uint64_t p_prev_cov_bb) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(active_mask);

    // get last bb from p_prev_cov_bb[thread_id], XOR it with current bb, and update the p_exec_cov_bb. 
    // Also store bb_id to p_prev_cov_bb[thread_id]
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x
          + threadIdx.y * blockDim.x * gridDim.x
          + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
          + threadIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y
          + blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;

    uint64_t* prev_cov_bb = (uint64_t*)p_prev_cov_bb;
    int new_bb_id = (MAP_SIZE/2) + ((prev_cov_bb[tid] ^ bb_id) % (MAP_SIZE/2));
    prev_cov_bb[tid] = bb_id >> 1;
   
    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        uint32_t* exec_cov_bb = (uint32_t*)p_exec_cov_bb;
        atomicAdd((unsigned int*)&exec_cov_bb[new_bb_id], 1);    
    }
}