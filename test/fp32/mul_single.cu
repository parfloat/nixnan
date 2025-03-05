#include <cstdio>
#include <cuda_runtime.h>
#include "helpers.cuh"
#include <cfloat>

// @min_sm 20

// sub{.rnd}{.ftz}{.sat}.f32  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_mul___(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul___sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul__ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul__ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rn__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rn.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rn__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rn.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rn_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rn_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rn.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rz__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rz__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rz_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rz.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rz_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rz.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rm__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rm.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rm__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rm.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rm_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rm.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rm_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rm.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rp__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rp.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rp__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rp.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rp_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rp.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_mul_rp_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("mul.rp.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}

typedef void (*kernel_t)(const float*, const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B, *C;
    kernel_t kernels[] = {test_mul___, test_mul___sat, test_mul__ftz_, test_mul__ftz_sat, test_mul_rn__,
                          test_mul_rn__sat, test_mul_rn_ftz_, test_mul_rn_ftz_sat, test_mul_rz__,
                          test_mul_rz__sat, test_mul_rz_ftz_, test_mul_rz_ftz_sat, test_mul_rm__,
                          test_mul_rm__sat, test_mul_rm_ftz_, test_mul_rm_ftz_sat, test_mul_rp__,
                          test_mul_rp__sat, test_mul_rp_ftz_, test_mul_rp_ftz_sat};  
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    cudaMallocManaged(&C, warpSize * sizeof(float));
    std::pair<float, float> error_pairs[] = {
        {INFINITY, -INFINITY},
        {FLT_MAX, FLT_MAX},
        {FLT_MIN, 0.5}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array_float(A, warpSize, a);
            fill_array_float(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
