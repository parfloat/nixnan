#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// rsqrt.approx{.ftz}.f32  d, a; // fast, approximate square root

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_rsqrt_approx_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rsqrt.approx.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rsqrt_approx_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}

typedef void (*kernel_t)(const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B;
    kernel_t kernels[] = {test_rsqrt_approx_, test_rsqrt_approx_ftz};
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    float vals[] = {-1.0, FLT_MIN, 0.0, -0.0, INFINITY};
    for (kernel_t k : kernels) {
        for (float val : vals) {
            fill_array_float(A, warpSize, val);
            k<<<1, 32>>>(A, B);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    return 0;
}
