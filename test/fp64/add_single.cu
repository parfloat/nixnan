#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// add{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_add_(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("add.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_add_rn(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("add.rn.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_add_rz(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("add.rz.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_add_rm(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("add.rm.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_add_rp(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("add.rp.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}

typedef void (*kernel_t)(const double*, const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B, *C;
    kernel_t kernels[] = {test_add_, test_add_rn, test_add_rz, test_add_rm, test_add_rp};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    cudaMallocManaged(&C, warpSize * sizeof(double));
    std::pair<float, float> error_pairs[] = {
        {INFINITY, -INFINITY},
        {FLT_MAX, FLT_MAX},
        {FLT_MIN, -FLT_MIN/2}};
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
