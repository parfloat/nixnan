/*
 * SPDX-FileCopyrightText: Modified code by NVIDIA CORPORATION & AFFILIATES.
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
 * cuFuzz NVBit Coverage Tool
 * Collects device-side edge coverage for AFL++ integration.
 *
 * Author: Mohamed Tarek (mtarek@nvidia.com)
 * Modified by: Mark S. Baranowski (mark.s.baranowski@gmail.com)
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <map>
#include <stdint.h>

/* every tool needs to include this once */
// #include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

#include <sys/shm.h> //AFL: giving NVBit access to AFL shared memory map

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* execution histogram of basic blocks */
#define MAP_SIZE 65536
#define MAX_BBS MAP_SIZE


// AFL hashing function and AFL unqiue constant 
#include "xxhash.h"
#define HASH_CONST 0xa5b35705

#define MAGIC_VALUE_START 1234 // For persistent mode
#define MAGIC_VALUE_END   5678 // For persistent mode


namespace cufuzz {
__managed__ uint32_t *exec_cov_bb; // no uint8_t atomiccas
__managed__ uint8_t *exec_cov_bb_quantized;
uint8_t* merged_cov;
uint8_t* trace_bits;
bool afl_set = false;
__managed__ uint64_t *prev_cov_bb; // added for edge coverage
#define MAX_THREADS 32000000 //10752

uint64_t total_bbs = 0;

typedef struct {
    uint32_t offset;
    std::string sass;
} instr_t;
std::vector<std::vector<instr_t>> bbs;

typedef struct {
    uint64_t pc;
    std::vector<int> bb_ids;
} kernel_t;
std::map<std::string, kernel_t> kernels;

/* global control variables for this tool */
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int active_from_start = 1;
bool mangled = false;
std::string outfilename = "";
bool serialize_grids = true;
int afl_persistent = 0;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0; // Added for edge-coverage 

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(verbose, "CUFUZZ_TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");
    GET_VAR_INT(serialize_grids, "SERIALIZE_GRIDS", 1, "Serialize grids");

    GET_VAR_STR(outfilename, "OUT_FILENAME",
                "Output file with execution histogram information");
    GET_VAR_INT(afl_persistent, "COV_PERSISTENT", 0, "Are we using AFL_PERSISTENT mode? 0:no, 1:yes");

    if (active_from_start == 0) {
        active_region = false;
    }

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    //AFL: mapping the AFL shared map into our address space 
    const char* shm_id_env = getenv("__AFL_SHM_ID");
    if (shm_id_env != NULL) {
        if (verbose) printf("CUFUZZ_COV: shm_id_env is: %s\n", shm_id_env);
        afl_set = true;
    } else {
        fprintf(stderr, "CUFUZZ_COV: AFL shared memory map not found, coverage will not be collected\n");
        afl_set = false;
        return;
    }
    int shm_id = atoi(shm_id_env);
    trace_bits = reinterpret_cast<uint8_t*>(shmat(shm_id, NULL, 0));
    if (trace_bits == (void *)-1) {
        if (verbose) fprintf(stderr, "CUFUZZ_COV: shmat failed\n");
        afl_set = false;
    }
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* get kernel name */
        std::string name = nvbit_get_func_name(ctx, f, mangled);

        /* if function already instrumented, return */
        if (kernels.find(name) != kernels.end()) {
            continue;
        }

        /* Get the static control flow graph of instruction */
        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) {
            printf(
                "Warning: Function %s is degenerated, we can't compute basic "
                "blocks statically",
                name.c_str());
        }

        if (verbose) {
            printf("Function %s\n", name.c_str());
            /* print */
            int cnt = 0;
            for (auto &bb : cfg.bbs) {
                printf("Basic block id %d - num instructions %ld\n", cnt++,
                       bb->instrs.size());
                for (auto &i : bb->instrs) {
                    i->print(" ");
                }
            }
        }

        if (verbose) {
            printf("inspecting %s - number basic blocks %ld\n", name.c_str(),
                   cfg.bbs.size());
        }
        total_bbs += cfg.bbs.size(); // AFL: Total number of bb represents the max number of hits in the coverage map

        uint64_t func_pc = nvbit_get_func_addr(ctx, f);

        kernels[name] = {func_pc, std::vector<int>()};

        /* Iterate on basic block and inject the first instruction */
        for (auto &bb : cfg.bbs) {
            int bb_id = bbs.size(); //AFL: bbs is a vector of all basic blocks
            if (verbose) printf("CUFUZZ_COV: bb_id: %d\n", bb_id);
            Instr *i = bb->instrs[0];
            /* inject device function */
            
            nvbit_insert_call(i, "record_coverage_edge_count", IPOINT_BEFORE);

            // use random id per basic block
            XXH64_hash_t fname_hash = XXH64(name.c_str(), name.length(), HASH_CONST);
            unsigned int bb_id_rand = (int)(((fname_hash << 5) + fname_hash) + bb_id);
            if (verbose) printf("CUFUZZ_COV: bb_id_rand: %u\n", bb_id_rand);
            nvbit_add_call_arg_const_val32(i, bb_id_rand);    

            /* AFL: add pointer to device-side coverage array */
            nvbit_add_call_arg_const_val64(i, (uint64_t)exec_cov_bb);
          
            /* add pointer to previous bb array */
            nvbit_add_call_arg_const_val64(i, (uint64_t)prev_cov_bb);

            if (verbose) {
                i->print("Inject count_instr before - ");
            }
            /* add basic block id used by this function */
            kernels[name].bb_ids.push_back(bb_id);
            /* add basic block to vector of all basic blocks */
            bbs.push_back(std::vector<instr_t>());
            /* add instructions to basic block */
            for (auto j : bb->instrs) {
                bbs[bb_id].push_back({j->getOffset(), j->getSass()});
            }
        }
    }
}

// AFL function for counting the number of bytes in the coverage map
uint32_t count_bytes(void *coverage_array, uint32_t size) {

  uint32_t *ptr = (uint32_t *)coverage_array;
  uint32_t  i = ((size + 3) >> 2);
  uint32_t  ret = 0;

  while (i--) {

    uint32_t v = *(ptr++);

    if (!v) { continue; }
    if (v & 0x000000ffU) { ++ret; }
    if (v & 0x0000ff00U) { ++ret; }
    if (v & 0x00ff0000U) { ++ret; }
    if (v & 0xff000000U) { ++ret; }

  }
  return ret;

}

// AFL function for printing the set bytes in the coverage map
void print_bytes(void *coverage_array, uint32_t size) {
    uint8_t *ptr = (uint8_t *)coverage_array;
    for (uint32_t i = 0; i < size; i++) {
        if (ptr[i] != 0) {
            printf("CUFUZZ_COV: Byte[%d]: %02X\n", i, ptr[i]);
        }
    }
}

uint32_t merge_coverage_byte_quant(void *host_cov, void* device_cov, void* new_cov) {

  uint8_t *ptr_dev = (uint8_t *)device_cov; 
  uint8_t *ptr_host = (uint8_t *)host_cov;
  uint8_t *ptr_new = (uint8_t *)new_cov;

  for (int ix = 0; ix < MAP_SIZE; ix++) {
    uint8_t hctr = ptr_host[ix];
    uint32_t dctr = ptr_dev[ix] + hctr;
    uint8_t dctr8 = dctr & 0xff;
    // Small change of dctr being zero since this is a 32-bit value.
    uint8_t ctr8 = dctr == 0 ? 0 : 
                    dctr8 == 0 ? 1 : dctr8;
    ptr_new[ix] = ctr8;
  }

  return 1;

}

// Given a non-zero Counter returns a number in the range [0,7].
uint8_t counterToByte(uint32_t counter) {
    // Returns a feature number by placing Counters into buckets as illustrated
    // below.
    //
    // Counter bucket: [0] [1] [2] [3-511] [512-4095] [4096-16383] [16384-65535] [65536+]
    // Feature number:  0   1   2     3         4           5             6         7
    uint8_t out_byte = 0;
    if (counter >= 65536) out_byte = 7;
    else if (counter >= 16384) out_byte = 6;
    else if (counter >= 4096) out_byte = 5;
    else if (counter >= 512) out_byte = 4;
    else if (counter >= 3) out_byte = 3;
    else if (counter >= 2) out_byte = 2;
    else if (counter >= 1) out_byte = 1;

    return out_byte;
}

// This function takes a device coverage map (32-bit entries) and maps it into 1-byte entry map that can be merged with trace_bits on the host 
void quantize_device_map(void *out_quantized, void *in_dev){

  uint32_t *ptr_in = (uint32_t *)in_dev; 
  uint8_t *ptr_out = (uint8_t *)out_quantized;

  for (int ix = 0; ix < MAP_SIZE; ix++) {
    uint32_t dctr = ptr_in[ix];
    uint8_t ctr8 = 0;
    ctr8 = counterToByte(dctr);
    ptr_out[ix] = ctr8;
  }

}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (!afl_set) {
        return;
    }
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        cuLaunch_params *p = (cuLaunch_params *)params;

        /* Check for cufuzz_notification_kernel (AFL persistent mode signaling) */
        std::string kernel_name = nvbit_get_func_name(ctx, p->f, 1);
        if (verbose) {printf("CUFUZZ_COV: DEBUG - Kernel name: '%s'\n", kernel_name.c_str());}
        if (kernel_name == "_Z26cufuzz_notification_kerneli" && afl_persistent && !is_exit) {
            /* Check if this is a notification kernel with specific parameters */
            if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params* p_kernel = (cuLaunchKernel_params*)params;
                
                /* Check if this is a 1x1 kernel launch (grid and block dimensions) */
                if (p_kernel->gridDimX == 1 && p_kernel->gridDimY == 1 && p_kernel->gridDimZ == 1 &&
                    p_kernel->blockDimX == 1 && p_kernel->blockDimY == 1 && p_kernel->blockDimZ == 1) {
                    
                    /* Get the first parameter which should be our magic value */
                    uint32_t magic_value = *((uint32_t*)p_kernel->kernelParams[0]);
                    
                    if (magic_value == MAGIC_VALUE_START) {
                        if (verbose) {printf("CUFUZZ_COV: cufuzz_notification_kernel(1234) detected - starting persistent loop iteration\n");}
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing host_cov:\n"); print_bytes((void*)trace_bits, MAP_SIZE);}
                        memset(trace_bits, 0, MAP_SIZE); // Reset AFL map at the beginning of the persistent loop
                        trace_bits[0] = 1;
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing host_cov:\n"); print_bytes((void*)trace_bits, MAP_SIZE);}
                    } else if (magic_value == MAGIC_VALUE_END) {
                        if (verbose) {printf("CUFUZZ_COV: cufuzz_notification_kernel(5678) detected - ending persistent loop iteration\n");}

                        // Merge device-side coverage with host-side coverage
                        quantize_device_map(exec_cov_bb_quantized, exec_cov_bb);            
                        uint32_t bytes_set_in_quantized_map = count_bytes((void*)exec_cov_bb_quantized, MAP_SIZE); 
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing exec_cov_bb_quantized:\n"); print_bytes((void*)exec_cov_bb_quantized, MAP_SIZE);}
                        XXH64_hash_t hash_quantized = XXH64(exec_cov_bb_quantized, MAP_SIZE, HASH_CONST);
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing host_cov:\n"); print_bytes((void*)trace_bits, MAP_SIZE);}
                        merge_coverage_byte_quant((void *)trace_bits, (void*) exec_cov_bb_quantized, (void*)merged_cov);
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing merged_cov:\n"); print_bytes((void*)merged_cov, MAP_SIZE);}
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: device_cov_quan: bytes_set_in_map: %d and hash: 0x%lx\n", bytes_set_in_quantized_map, hash_quantized);}
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV:   host_cov: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)trace_bits, MAP_SIZE), XXH64(trace_bits, MAP_SIZE, HASH_CONST));}
                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: merged_cov: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)merged_cov, MAP_SIZE), XXH64(merged_cov, MAP_SIZE, HASH_CONST));}

                        memcpy(trace_bits, merged_cov, MAP_SIZE);
                        memset(exec_cov_bb, 0, sizeof(uint32_t) * MAX_BBS); // Clear before next AFL_LOOP iteration
                        memset(exec_cov_bb_quantized, 0, MAP_SIZE); // Clear before next AFL_LOOP iteration

                        if (verbose){ fprintf(stdout, "CUFUZZ_COV: trace_bits: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)trace_bits, MAP_SIZE), XXH64(trace_bits, MAP_SIZE, HASH_CONST));}
                    }
                }
            }
            /* Return early for notification kernels to avoid normal processing */
            return;
        }

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, p->f);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }
            /* set grid launch id at launch time */
            nvbit_set_at_launch(ctx, p->f, (uint64_t)&grid_launch_id);
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                uint64_t num_threads = p->gridDimX * p->gridDimY * p->gridDimZ * p->blockDimX * p->blockDimY * p->blockDimZ;
                if (verbose){
                    printf(
                    "Entering: CTX 0x%016lx - LAUNCH "
                    "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                    "- block size %d,%d,%d - cuda stream "
                    "id %ld - threads: %ld\n",
                    (uint64_t)ctx, nvbit_get_func_name(ctx, p->f, mangled), grid_launch_id, p->gridDimX,
                    p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                    p->blockDimZ, (uint64_t)p->hStream, num_threads);
                }
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, p->f, true);
            } else {
                nvbit_enable_instrumented(ctx, p->f, false);
            }
            if (verbose) {
                int num_ctas = 0;
                if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                    cbid == API_CUDA_cuLaunchKernel) {
                    cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                    num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                }
                printf("GRID %d - %s - #thread-blocks %d\n", kernel_id++,
                       nvbit_get_func_name(ctx, p->f, mangled), num_ctas);
            }
            pthread_mutex_unlock(&mutex);
        } else {
            if (serialize_grids) {
                CUDA_SAFECALL(cudaDeviceSynchronize());
            }
            // Added for edge-coverage, we want to clean the state after edge info is collected 
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                uint64_t num_threads = p->gridDimX * p->gridDimY * p->gridDimZ * p->blockDimX * p->blockDimY * p->blockDimZ;
                if (verbose){
                    printf(
                    "Exiting: CTX 0x%016lx - LAUNCH "
                    "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                    "- block size %d,%d,%d - cuda stream "
                    "id %ld - threads: %ld\n",
                    (uint64_t)ctx, nvbit_get_func_name(ctx, p->f, mangled), grid_launch_id, p->gridDimX,
                    p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                    p->blockDimZ, (uint64_t)p->hStream, num_threads);   
                }
            }
            /* increment grid launch id for next launch */
            grid_launch_id++;
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    } 
}

int cnt_ctx = 0;

void nvbit_tool_init(CUcontext ctx) {
    if (!afl_set) {
        return;
    }
    pthread_mutex_lock(&mutex);
    if (cnt_ctx == 0) {
        // Device-side AFL coverage map
        cudaMallocManaged(&exec_cov_bb, sizeof(uint32_t) * MAX_BBS);
        memset(exec_cov_bb, 0, sizeof(uint32_t) * MAX_BBS);

        cudaMallocManaged(&exec_cov_bb_quantized, MAP_SIZE);
        memset(exec_cov_bb_quantized, 0, MAP_SIZE);

        merged_cov = (uint8_t*) malloc(MAP_SIZE);
        memset(merged_cov, 0, MAP_SIZE);
        if (verbose){ fprintf(stdout, "nvbit_tool_init: CUFUZZ_COV: Printing host_cov:\n"); print_bytes((void*)trace_bits, MAP_SIZE);}

        // Allocate storage for the edge coverage (previous as current is already available for each thread) 
        // TODO: need to be computed per kernel rather than globally 
        cudaMallocManaged(&prev_cov_bb, sizeof(uint64_t) * MAX_THREADS);
        memset(prev_cov_bb, 0, sizeof(uint64_t) * MAX_THREADS);
    }
    cnt_ctx++;
    pthread_mutex_unlock(&mutex);
}


void nvbit_at_ctx_term(CUcontext ctx) {
    if (!afl_set) {
        return;
    }
    pthread_mutex_lock(&mutex);
    cnt_ctx--;
    if (cnt_ctx == 0) {
        if(afl_persistent == 0){
            //AFL: printing some stats for coverage update
            quantize_device_map(exec_cov_bb_quantized, exec_cov_bb);            
            uint32_t bytes_set_in_quantized_map = count_bytes((void*)exec_cov_bb_quantized, MAP_SIZE); 
            if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing exec_cov_bb_quantized:\n"); print_bytes((void*)exec_cov_bb_quantized, MAP_SIZE);}
            XXH64_hash_t hash_quantized = XXH64(exec_cov_bb_quantized, MAP_SIZE, HASH_CONST);
            if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing host_cov:\n"); print_bytes((void*)trace_bits, MAP_SIZE);}
            merge_coverage_byte_quant((void *)trace_bits, (void*) exec_cov_bb_quantized, (void*)merged_cov);
            if (verbose){ fprintf(stdout, "CUFUZZ_COV: Printing merged_cov:\n"); print_bytes((void*)merged_cov, MAP_SIZE);}
            if (verbose){ fprintf(stdout, "CUFUZZ_COV: device_cov_quan: bytes_set_in_map: %d and hash: 0x%lx\n", bytes_set_in_quantized_map, hash_quantized);}
            if (verbose){ fprintf(stdout, "CUFUZZ_COV:   host_cov: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)trace_bits, MAP_SIZE), XXH64(trace_bits, MAP_SIZE, HASH_CONST));}
            if (verbose){ fprintf(stdout, "CUFUZZ_COV: merged_cov: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)merged_cov, MAP_SIZE), XXH64(merged_cov, MAP_SIZE, HASH_CONST));}
            memcpy(trace_bits, merged_cov, MAP_SIZE);
		    if (verbose){ fprintf(stdout, "CUFUZZ_COV: trace_bits: bytes_set_in_map: %d and hash: 0x%lx\n", count_bytes((void*)trace_bits, MAP_SIZE), XXH64(trace_bits, MAP_SIZE, HASH_CONST));}
        }

        // Delete memory 
        cudaError_t cudaStat;
        // Device-side AFL coverage map
        cudaStat = cudaFree(exec_cov_bb);  if (cudaStat != cudaSuccess) fprintf (stdout, "CUFUZZ_COV: cudaFree exec_cov_bb Failed");
        cudaStat = cudaFree(exec_cov_bb_quantized);  if (cudaStat != cudaSuccess) fprintf (stdout, "CUFUZZ_COV: cudaFree exec_cov_bb_quantized Failed");
        free(merged_cov);

        cudaStat = cudaFree(prev_cov_bb); 
        if (cudaStat != cudaSuccess) printf ("CUFUZZ_COV: cudaFree prev_cov_bb Failed\n");
    }
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_term() {  
    return;
}
} // namespace cufuzz