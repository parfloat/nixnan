#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>
#include <cuda_fp16.h>

// @min_sm 20

// sub{.rnd}{.ftz}{.sat}.f16   d, a, b;
// sub{.rnd}{.ftz}{.sat}.f16x2 d, a, b;

// .rnd = { .rn };

__global__ void test_fma___(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma___sat(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.sat.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma__ftz_(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.ftz.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma__ftz_sat(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.ftz.sat.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma_rn__(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.rn.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma_rn__sat(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.rn.sat.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma_rn_ftz_(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.rn.ftz.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}
__global__ void test_fma_rn_ftz_sat(const half* a, const half* b, half* c) {
    int idx = threadIdx.x;
    short out;
    asm ("sub.rn.ftz.sat.f16 %0, %1, %2;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])));
    c[idx] = s2h(out);
}

typedef void (*kernel_t)(const half*, const half*, half*);
int main() {
    int warpSize = 32;
    half *A, *B, *C;
    kernel_t kernels[] = {test_fma___, test_fma___sat, test_fma__ftz_, test_fma__ftz_sat,
                          test_fma_rn__, test_fma_rn__sat, test_fma_rn_ftz_,
                          test_fma_rn_ftz_sat};
    cudaMallocManaged(&A, warpSize * sizeof(half));
    cudaMallocManaged(&B, warpSize * sizeof(half));
    cudaMallocManaged(&C, warpSize * sizeof(half));
    std::pair<half, half> error_pairs[] = {
        {INFINITY, INFINITY},
        {HLF_MAX, -HLF_MAX},
        {HLF_MIN, HLF_MIN/2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array_half(A, warpSize, a);
            fill_array_half(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
