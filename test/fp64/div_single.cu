#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// div{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_div_rn(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("div.rn.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_div_rz(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("div.rz.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_div_rm(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("div.rm.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}
__global__ void test_div_rp(const double* a, const double* b, double* c) {
    int idx = threadIdx.x;
    asm ("div.rp.f64 %0, %1, %2;" : "=d"(c[idx]) : "d"(a[idx]), "d"(b[idx]));
}

typedef void (*kernel_t)(const double*, const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B, *C;
    kernel_t kernels[] = {test_div_rn, test_div_rz, test_div_rm, test_div_rp};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    cudaMallocManaged(&C, warpSize * sizeof(double));
    std::pair<double, double> error_pairs[] = {
        {1.0, 0.0},
        {INFINITY, 0.0},
        {DBL_MIN, 2.0}};
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
