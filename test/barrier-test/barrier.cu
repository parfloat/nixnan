#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void simple_barrier_kernel(float* data) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x;
    int global_idx = block_offset + tid;

    // Phase 1: even-numbered threads generate NaN via sqrt(-1)
    if (tid % 2 == 0) {
        data[global_idx] = sqrtf(-1.0f);  // Generate NaN
    } else {
        data[global_idx] = (float)tid;
    }

    // Synchronize all threads in the block
    __syncthreads();

    // Phase 2: each thread reads from right neighbor (within block)
    int neighbor_tid = (tid + 1) % blockDim.x;
    int neighbor_idx = block_offset + neighbor_tid;
    float neighbor_value = data[neighbor_idx];
    data[global_idx] = (float)tid + neighbor_value;

    // Synchronize all threads in the block
    __syncthreads();

    // Phase 3: odd threads generate NaN via sqrt(-1)
    if (tid % 2 == 1) {
        data[global_idx] = sqrtf(-1.0f);  // Generate NaN
    }

    // Synchronize all threads in the block
    __syncthreads();

    // Phase 4: each thread reads from left neighbor (within block)
    int left_neighbor_tid = (tid - 1 + blockDim.x) % blockDim.x;
    int left_neighbor_idx = block_offset + left_neighbor_tid;
    float left_neighbor_value = data[left_neighbor_idx];
    data[global_idx] = left_neighbor_value;
}

int main() {
    const int N = 32; // Number of threads per block
    const int NUM_BLOCKS = 4; // Number of blocks

    // Allocate device memory for all blocks
    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, NUM_BLOCKS * N * sizeof(float)));

    // Launch kernel with 4 blocks of N threads each
    simple_barrier_kernel<<<NUM_BLOCKS, N>>>(d_data);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results back
    float h_data[NUM_BLOCKS * N];
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, NUM_BLOCKS * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_data));

    // Verify results (first 2 elements from each block)
    std::cout << "Kernel executed successfully with " << NUM_BLOCKS << " blocks." << std::endl;
    std::cout << "Results (first 2 elements from each block):" << std::endl;
    for (int block = 0; block < NUM_BLOCKS; block++) {
        std::cout << "Block " << block << ":" << std::endl;
        for (int i = 0; i < 2; i++) {
            int idx = block * N + i;
            std::cout << "  data[" << idx << "] = " << h_data[idx] << std::endl;
        }
    }

    return 0;
}
