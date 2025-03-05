#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// rcp.approx{.ftz}.f32  d, a;  // fast, approximate reciprocal
// rcp.rnd{.ftz}.f32     d, a;  // IEEE 754 compliant rounding

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_rcp_approx_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.approx.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_approx_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.approx.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rn_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rn.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rn_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rn.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rz_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rz_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rz.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rm_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rm.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rm_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rm.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rp_(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rp.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}
__global__ void test_rcp_rp_ftz(const float* a, float* b) {
    int idx = threadIdx.x;
    asm ("rcp.rp.ftz.f32 %0, %1;" : "=f"(b[idx]) : "f"(a[idx]));
}

typedef void (*kernel_t)(const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B;
    kernel_t kernels[] = {test_rcp_approx_, test_rcp_approx_ftz, test_rcp_rn_, test_rcp_rn_ftz, test_rcp_rz_,
                          test_rcp_rz_ftz, test_rcp_rm_, test_rcp_rm_ftz, test_rcp_rp_, test_rcp_rp_ftz};
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    float vals[] = {0.0, INFINITY, -0.0, NAN, FLT_MIN, FLT_MAX};
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
