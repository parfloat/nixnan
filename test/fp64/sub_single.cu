#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// sub{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_sub_(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("sub.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_sub_rn(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("sub.rn.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_sub_rz(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("sub.rz.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_sub_rm(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("sub.rm.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_sub_rp(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("sub.rp.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}

typedef void (*kernel_t)(const double*, const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B, *C;
    kernel_t kernels[] = {test_sub_, test_sub_rn, test_sub_rz, test_sub_rm, test_sub_rp};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    cudaMallocManaged(&C, warpSize * sizeof(double));
    std::pair<double, double> error_pairs[] = {
        {INFINITY, INFINITY},
        {DBL_MAX, -DBL_MAX},
        {DBL_MIN, DBL_MIN/2}};
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
