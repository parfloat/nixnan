#include <cstdio>
#include <cuda_runtime.h>
#include "helpers.hu"
#include <cfloat>

// @min_sm 20

// add{.rnd}{.ftz}{.sat}.f32  d, a, b;
// add{.rnd}{.ftz}.f32x2      d, a, b;
// add{.rnd}.f64              d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test___(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test___sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test__ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test__ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rn__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rn.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rn__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rn.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rn_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rn.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rn_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rn.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rz__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rz__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rz_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rz.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rz_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rz.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rm__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rm.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rm__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rm.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rm_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rm.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rm_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rm.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rp__(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rp.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rp__sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rp.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rp_ftz_(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rp.ftz.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}
__global__ void test_rp_ftz_sat(const float* a, const float* b, float* c) {
    int idx = threadIdx.x;
    asm ("add.rp.ftz.sat.f32 %0, %1, %2;" : "=f"(c[idx]) : "f"(a[idx]), "f"(b[idx]));
}

typedef void (*kernel_t)(const float*, const float*, float*);
int main() {
    int warpSize = 32;
    float *A, *B, *C;
    kernel_t kernels[] = {test___, test___sat, test__ftz_, test__ftz_sat, test_rn__,
                          test_rn__sat, test_rn_ftz_, test_rn_ftz_sat, test_rz__,
                          test_rz__sat, test_rz_ftz_, test_rz_ftz_sat, test_rm__,
                          test_rm__sat, test_rm_ftz_, test_rm_ftz_sat, test_rp__,
                          test_rp__sat, test_rp_ftz_, test_rp_ftz_sat};
    cudaMallocManaged(&A, warpSize * sizeof(float));
    cudaMallocManaged(&B, warpSize * sizeof(float));
    cudaMallocManaged(&C, warpSize * sizeof(float));
    std::pair<float, float> error_pairs[] = {
        {INFINITY, -INFINITY},
        {FLT_MAX, FLT_MAX},
        {FLT_MIN, -FLT_MIN/2}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array(A, warpSize, a);
            fill_array(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
