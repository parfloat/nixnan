#include <cstdio>
#include <cuda_runtime.h>
#include "helpers.cuh"
#include <cfloat>

// @min_sm 20

// add{.rnd}{.ftz}{.sat}.f32  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_div_approx_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.approx.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_approx_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.approx.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_full_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.full.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_full_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.full.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rn_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rn.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rn_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rn.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rz_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rz.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rm_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rm.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rm_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rm.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rp_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rp.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_div_rp_ftz(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("div.rp.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}

typedef void (*kernel_t)(const float*, const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B, *C;
    kernel_t kernels[] = {test_div_approx_, test_div_approx_ftz, test_div_full_, test_div_full_ftz,
                          test_div_rn_, test_div_rn_ftz, test_div_rz_, test_div_rz_ftz,
                          test_div_rm_, test_div_rm_ftz,test_div_rp_, test_div_rp_ftz};
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    cudaMallocManaged(&C, warpSize * sizeof(float));
    std::pair<float, float> error_pairs[] = {
        {1.0, 0.0},
        {1.0, -0.0},
        {FLT_MAX, 0.5},
        {FLT_MIN, 2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array_float(A, warpSize, a);
            fill_array_float(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
            for (int i = 0; i < warpSize; i++) {
                printf("%e / %e = %e\n", A[i], B[i], C[i]);
            }
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
