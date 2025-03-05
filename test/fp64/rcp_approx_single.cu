#include <cstdio>
#include <cuda_runtime.h>
#include "../helpers.cuh"
#include <cfloat>

// @min_sm 20

// rcp{.rnd}.f64  d, a, b;

// .rnd = { .rn, .rz, .rm, .rp };

__global__ void test_approx_ftz(const double* a, double* b) {
    int idx = threadIdx.x;
    asm ("rcp.approx.ftz.f64 %0, %1;" : "=d"(b[idx]) : "d"(a[idx]));
}

typedef void (*kernel_t)(const double*, double*);
int main() {
    int warpSize = 32;
    double *A, *B;
    kernel_t kernels[] = {test_approx_ftz};
    cudaMallocManaged(&A, warpSize * sizeof(double));
    cudaMallocManaged(&B, warpSize * sizeof(double));
    double error_pairs[] = { 0.0, INFINITY, -INFINITY, DBL_MIN, DBL_MAX };
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
