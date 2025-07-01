#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>
#include <cuda_fp16.h>

// @min_sm 20

// sub{.rnd}{.ftz}{.sat}.f16x2   d, a, b;
// sub{.rnd}{.ftz}{.sat}.f16x2x2 d, a, b;

// .rnd = { .rn };

__global__ void test_fma___(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma___sat(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.sat.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma__ftz_(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.ftz.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma__ftz_sat(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.ftz.sat.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma_rn__(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma_rn__sat(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.rn.sat.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma_rn_ftz_(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.rn.ftz.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}
__global__ void test_fma_rn_ftz_sat(const half2* a, const half2* b, half2* c) {
    int idx = threadIdx.x;
    unsigned int out;
    asm ("sub.rn.ftz.sat.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h22u(a[idx])), "r"(h22u(b[idx])));
    c[idx] = u2h2(out);
}

typedef void (*kernel_t)(const half2*, const half2*, half2*);
int main() {
    int warpSize = 32;
    half2 *A, *B, *C;
    kernel_t kernels[] = {test_fma___, test_fma___sat, test_fma__ftz_, test_fma__ftz_sat,
                          test_fma_rn__, test_fma_rn__sat, test_fma_rn_ftz_,
                          test_fma_rn_ftz_sat};
    cudaMallocManaged(&A, warpSize * sizeof(half2));
    cudaMallocManaged(&B, warpSize * sizeof(half2));
    cudaMallocManaged(&C, warpSize * sizeof(half2));
    std::pair<half, half> error_pairs[] = {
        {INFINITY, -INFINITY},
        {HLF_MAX, HLF_MAX},
        {HLF_MIN, -HLF_MIN/2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array_half2(A, warpSize, a);
            fill_array_half2(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
