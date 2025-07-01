#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>
#include <cuda_fp16.h>
#include <tuple>

// @min_sm 20

// fma.rnd{.ftz}{.sat}.f16     d, a, b, c;
// fma.rnd{.ftz}{.sat}.f16x2   d, a, b, c;
// fma.rnd{.ftz}.relu.f16      d, a, b, c;
// fma.rnd{.ftz}.relu.f16x2    d, a, b, c;
// fma.rnd{.relu}.bf16         d, a, b, c;
// fma.rnd{.relu}.bf16x2       d, a, b, c;
// fma.rnd.oob.{relu}.type     d, a, b, c;

// .rnd = { .rn };

__global__ void test_fma_rn__(const half* a, const half* b, const half* c, half* d) {
    int idx = threadIdx.x;
    short out;
    asm ("fma.rn.f16 %0, %1, %2, %3;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])), "h"(h2s(c[idx])));
    d[idx] = s2h(out);
}
__global__ void test_fma_rn__sat(const half* a, const half* b, const half* c, half* d) {
    int idx = threadIdx.x;
    short out;
    asm ("fma.rn.sat.f16 %0, %1, %2, %3;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])), "h"(h2s(c[idx])));
    d[idx] = s2h(out);
}
__global__ void test_fma_rn_ftz_(const half* a, const half* b, const half* c, half* d) {
    int idx = threadIdx.x;
    short out;
    asm ("fma.rn.ftz.f16 %0, %1, %2, %3;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])), "h"(h2s(c[idx])));
    d[idx] = s2h(out);
}
__global__ void test_fma_rn_ftz_sat(const half* a, const half* b, const half* c, half* d) {
    int idx = threadIdx.x;
    short out;
    asm ("fma.rn.ftz.sat.f16 %0, %1, %2, %3;" : "=h"(out) : "h"(h2s(a[idx])), "h"(h2s(b[idx])), "h"(h2s(c[idx])));
    d[idx] = s2h(out);
}

typedef void (*kernel_t)(const half*, const half*, const half*, half*);
int main() {
    int warpSize = 32;
    half *A, *B, *C, *D;
    kernel_t kernels[] = {test_fma_rn__, test_fma_rn__sat, test_fma_rn_ftz_, test_fma_rn_ftz_sat};
    cudaMallocManaged(&A, warpSize * sizeof(half));
    cudaMallocManaged(&B, warpSize * sizeof(half));
    cudaMallocManaged(&C, warpSize * sizeof(half));
    cudaMallocManaged(&D, warpSize * sizeof(half));
    std::tuple<half, half, half> error_pairs[] = {
        {HLF_MAX, HLF_MAX, 0},
        {INFINITY, 1.0, -INFINITY},
        {HLF_MIN, 0.5, 0},
        {HLF_MIN, 1.0, -HLF_MIN/2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b, c] : error_pairs) {
            fill_array_half(A, warpSize, a);
            fill_array_half(B, warpSize, b);
            fill_array_half(C, warpSize, c);
            k<<<1, 32>>>(A, B, C, D);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    return 0;
}
