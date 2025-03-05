#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// sqrt.approx{.ftz}.f32  d, a; // fast, approximate square root
// sqrt.rnd{.ftz}.f32     d, a; // IEEE 754 compliant rounding

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_sqrt_approx_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.approx.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_approx_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.approx.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rn_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rn.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rn_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rn.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rz_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rz_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rz.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rm_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rm.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rm_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rm.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rp_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rp.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_sqrt_rp_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rp.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}

typedef void (*kernel_t)(const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B;
    kernel_t kernels[] = {test_sqrt_approx_, test_sqrt_approx_ftz, test_sqrt_rn_, test_sqrt_rn_ftz, test_sqrt_rz_,
                          test_sqrt_rz_ftz, test_sqrt_rm_, test_sqrt_rm_ftz, test_sqrt_rp_, test_sqrt_rp_ftz};
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
