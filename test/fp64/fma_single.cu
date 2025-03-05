#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>
#include <tuple>

// @min_sm 20

// fma{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_fma_rn(const double* a, const double* b, const double* c, double* d) {
    int idx = threadIdx.x;
    asm ("fma.rn.f64 %0, %1, %2, %3;" : "=d"(d[idx]) : "d"(a[idx]), "d"(b[idx]), "d"(c[idx]));
}
__global__ void test_fma_rz(const double* a, const double* b, const double* c, double* d) {
    int idx = threadIdx.x;
    asm ("fma.rz.f64 %0, %1, %2, %3;" : "=d"(d[idx]) : "d"(a[idx]), "d"(b[idx]), "d"(c[idx]));
}
__global__ void test_fma_rm(const double* a, const double* b, const double* c, double* d) {
    int idx = threadIdx.x;
    asm ("fma.rm.f64 %0, %1, %2, %3;" : "=d"(d[idx]) : "d"(a[idx]), "d"(b[idx]), "d"(c[idx]));
}
__global__ void test_fma_rp(const double* a, const double* b, const double* c, double* d) {
    int idx = threadIdx.x;
    asm ("fma.rp.f64 %0, %1, %2, %3;" : "=d"(d[idx]) : "d"(a[idx]), "d"(b[idx]), "d"(c[idx]));
}

typedef void (*kernel_t)(const double*, const double*, const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B, *C, *D;
    kernel_t kernels[] = {test_fma_rn, test_fma_rz, test_fma_rm, test_fma_rp};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    cudaMallocManaged(&C, warpSize * sizeof(double));
    cudaMallocManaged(&D, warpSize * sizeof(double));
    std::tuple<double, double, double> error_pairs[] = {
        {DBL_MAX, DBL_MAX, 0},
        {INFINITY, 1.0, -INFINITY},
        {DBL_MIN, 0.5, 0},
        {DBL_MIN, 1.0, -DBL_MIN/2.0}};
    for (kernel_t k : kernels) {
        for (auto [a, b, c] : error_pairs) {
            fill_array_double(A, warpSize, a);
            fill_array_double(B, warpSize, b);
            fill_array_double(C, warpSize, c);
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
