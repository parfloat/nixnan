#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// mul{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_mul_(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("mul.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_mul_rn(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("mul.rn.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_mul_rz(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("mul.rz.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_mul_rm(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("mul.rm.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_mul_rp(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("mul.rp.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}

typedef void (*kernel_t)(const double*, const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B, *C;
    kernel_t kernels[] = {test_mul_, test_mul_rn, test_mul_rz, test_mul_rm, test_mul_rp};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    cudaMallocManaged(&C, warpSize * sizeof(double));
    std::pair<float, float> error_pairs[] = {
        {INFINITY, -INFINITY},
        {DBL_MAX, DBL_MAX},
        {DBL_MIN, 0.5}};
    for (kernel_t k : kernels) {
        for (auto [a, b] : error_pairs) {
            fill_array_double(A, warpSize, a);
            fill_array_double(B, warpSize, b);
            k<<<1, 32>>>(A, B, C);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
