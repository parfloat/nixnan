#include <cstdio>
#include <cuda_runtime.h>
#include "helpers.cuh"
#include <cfloat>
#include <tuple>

// @min_sm 20

// fma.rnd{.ftz}{.sat}.f32  d, a, b, c;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_fma_rn__(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rn__sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rn_ftz_(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rn_ftz_sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rn.ftz.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rz__(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rz__sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rz.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rz_ftz_(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rz.ftz.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rz_ftz_sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rz.ftz.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rm__(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rm.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rm__sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rm.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rm_ftz_(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rm_ftz_sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rm.ftz.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rp__(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rp.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rp__sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rp.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rp_ftz_(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rp.ftz.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}
__global__ void test_fma_rp_ftz_sat(const float* a, const float* b, const float* c, float* d) {
    int idx = threadIdx.x;
    asm ("fma.rp.ftz.sat.f32 %0, %1, %2, %3;" : "=f"(d[idx]) : "f"(a[idx]), "f"(b[idx]), "f"(c[idx]));
}

typedef void (*kernel_t)(const float*, const float*, const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B, *C, *D;
    kernel_t kernels[] = {test_fma_rn__, test_fma_rn__sat, test_fma_rn_ftz_, test_fma_rn_ftz_sat,
                          test_fma_rz__, test_fma_rz__sat, test_fma_rz_ftz_, test_fma_rz_ftz_sat,
                          test_fma_rm__, test_fma_rm__sat, test_fma_rm_ftz_, test_fma_rm_ftz_sat,
                          test_fma_rp__, test_fma_rp__sat, test_fma_rp_ftz_, test_fma_rp_ftz_sat};  
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    cudaMallocManaged(&C, warpSize * sizeof(float));
    cudaMallocManaged(&D, warpSize * sizeof(float));
    std::tuple<float, float, float> error_pairs[] = {
        {FLT_MAX, FLT_MAX, 0},
        {INFINITY, 1.0, -INFINITY},
        {FLT_MIN, 0.5, 0},
        {FLT_MIN, 1.0, -FLT_MIN/2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b, c] : error_pairs) {
            fill_array_float(A, warpSize, a);
            fill_array_float(B, warpSize, b);
            fill_array_float(C, warpSize, c);
            k<<<1, 32>>>(A, B, C, D);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}