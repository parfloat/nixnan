#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// rsqrt.approx{.ftz}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_rsqrt_approx(const double* a, double* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rn.f64 %0, %1;" : "=d"(b[idx]) : "d"(a[idx]));
}
__global__ void test_rsqrt_approx_ftz(const double* a, double* b) {
    int idx = threadIdx.x;
    asm ("sqrt.rn.f64 %0, %1;" : "=d"(b[idx]) : "d"(a[idx]));
}


typedef void (*kernel_t)(const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B;
    kernel_t kernels[] = {test_rsqrt_approx, test_rsqrt_approx_ftz};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    double error_pairs[] = { 0.0, INFINITY, -1.0, -0.0, -INFINITY, -DBL_MIN/2 };
    for (kernel_t k : kernels) {
        for (auto a : error_pairs) {
            fill_array_double(A, warpSize, a);
            k<<<1, 32>>>(A, B);
            cudaDeviceSynchronize();
        }
    }
    cudaFree(A);
    cudaFree(B);
    return 0;
}
