#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Macro to check for CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel for double
__global__ void set_values_double(double* data, double value) {
    unsigned int idx = threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = value+value;
    } else {
        data[idx] = -value-value;
    }
}

// Kernel for float
__global__ void set_values_float(float* data, float value) {
    unsigned int idx = threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = value+value;
    } else {
        data[idx] = -value-value;
    }
}

// Kernel for half
__global__ void set_values_half(__half* data, __half value) {
    unsigned int idx = threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = value+value;
    } else {
        data[idx] = -value-value;
    }
}

// Kernel for bfloat16
__global__ void set_values_bfloat16(__nv_bfloat16* data, __nv_bfloat16 value) {
    unsigned int idx = threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = value+value;
    } else {
        data[idx] = -value-value;
    }
}

int main() {
    const int N = 256; // Number of elements, must be <= 1024 for one block

    // Allocate device memory
    double* d_double_data;
    float* d_float_data;
    __half* d_half_data;
    __nv_bfloat16* d_bfloat16_data;

    CHECK_CUDA_ERROR(cudaMalloc(&d_double_data, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_float_data, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_half_data, N * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bfloat16_data, N * sizeof(__nv_bfloat16)));

    // Define values to be passed to kernels
    const double val_double = INFINITY;
    const float val_float = INFINITY;
    const __half val_half = __float2half(INFINITY);
    const __nv_bfloat16 val_bfloat16 = __float2bfloat16(INFINITY);

    // Launch kernels
    set_values_double<<<1, N>>>(d_double_data, val_double);
    CHECK_CUDA_ERROR(cudaGetLastError());
    set_values_float<<<1, N>>>(d_float_data, val_float);
    CHECK_CUDA_ERROR(cudaGetLastError());
    set_values_half<<<1, N>>>(d_half_data, val_half);
    CHECK_CUDA_ERROR(cudaGetLastError());
    set_values_bfloat16<<<1, N>>>(d_bfloat16_data, val_bfloat16);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Synchronize to ensure kernels have completed
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate host memory to verify results
    std::vector<double> h_double_data(N);
    std::vector<float> h_float_data(N);
    std::vector<__half> h_half_data(N);
    std::vector<__nv_bfloat16> h_bfloat16_data(N);

    // Copy data back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_double_data.data(), d_double_data, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_float_data.data(), d_float_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_half_data.data(), d_half_data, N * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_bfloat16_data.data(), d_bfloat16_data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_double_data));
    CHECK_CUDA_ERROR(cudaFree(d_float_data));
    CHECK_CUDA_ERROR(cudaFree(d_half_data));
    CHECK_CUDA_ERROR(cudaFree(d_bfloat16_data));

    std::cout << "Kernels executed successfully." << std::endl;
    std::cout << "Verification (first 4 elements):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        float f_half = __half2float(h_half_data[i]);
        float f_bfloat = __bfloat162float(h_bfloat16_data[i]);
        std::cout << "Index " << i << ":" << std::endl;
        std::cout << "  double:  " << h_double_data[i] << std::endl;
        std::cout << "  float:   " << h_float_data[i] << std::endl;
        std::cout << "  half:    " << f_half << std::endl;
        std::cout << "  bfloat16:" << f_bfloat << std::endl;
    }

    return 0;
}
