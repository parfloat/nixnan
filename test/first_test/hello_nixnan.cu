// hello_nixnan.cu -- a tiny program that raises floating-point exceptions on purpose.
//   build:  nvcc -lineinfo -arch=native -o hello_nixnan hello_nixnan.cu
//   run:    LD_PRELOAD=../../nixnan.so ./hello_nixnan      (nixnan prints its report on exit)
// @min_sm 80
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

// Scalar kernel: exceptions in FP64, FP32, FP16 and BF16, each result stored to global memory (STG).
// x = 0, half = 0.5 and inf come from memory, so the compiler cannot fold the arithmetic away.
__global__ void hello(const double* in, double* d, float* f, half2* h, __nv_bfloat162* b) {
    double x = in[0], half_ = in[1], inf = in[2];       // 0.0, 0.5, +Inf
    d[0] = 1.0 / x;                                     // FP64: +Inf, division by zero
    d[1] = inf * x;                                     // FP64: Inf * 0 = NaN -> STG.E.64 of a NaN
    float fx = (float)x;
    f[0] = 1.0f / fx;                                   // FP32: +Inf, division by zero
    f[1] = (float)inf * fx;                              // FP32: Inf * 0 = NaN -> STG.E of a NaN
    f[2] = 1.17549435e-38f * (float)half_;              // FP32: subnormal (FLT_MIN * 0.5)
    half2 big = __float2half2_rn(65504.0f);             // FP16 max
    h[0] = __hmul2(big, big);                           // FP16: +Inf (overflow)
    h[1] = __hmul2(h[0], __float2half2_rn(fx));         // FP16: Inf * 0 = NaN -> STG.E of a NaN
    __nv_bfloat162 bbig = __float2bfloat162_rn(3.0e38f);
    b[0] = __hmul2(bbig, bbig);                         // BF16: +Inf (overflow)
    b[1] = __hmul2(b[0], __float2bfloat162_rn(fx));     // BF16: Inf * 0 = NaN -> STG.E of a NaN
}

// Tensor core: one 16x16x16 FP16 multiply with FP32 accumulate (HMMA.16816.F32).
// A[0,0] = Inf makes Inf * 0 = NaN across row 0 of the result.
__global__ void tc(const half* A, const half* B, float* D) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(D, c, 16, wmma::mem_row_major);
}

int main() {
    double *in, *d; float *f, *D; half2 *h; __nv_bfloat162 *b; half *A, *B;
    cudaMallocManaged(&in, 3 * sizeof(double)); cudaMallocManaged(&d, 2 * sizeof(double));
    cudaMallocManaged(&f, 3 * sizeof(float));   cudaMallocManaged(&h, 2 * sizeof(half2));
    cudaMallocManaged(&b, 2 * sizeof(__nv_bfloat162));
    cudaMallocManaged(&A, 256 * sizeof(half)); cudaMallocManaged(&B, 256 * sizeof(half));
    cudaMallocManaged(&D, 256 * sizeof(float));
    in[0] = 0.0; in[1] = 0.5; in[2] = INFINITY;
    for (int i = 0; i < 256; i++) { A[i] = __float2half(i % 17 == 0); B[i] = __float2half(i % 17 == 0); }
    A[0] = __float2half(INFINITY);

    hello<<<1, 1>>>(in, d, f, h, b);
    tc<<<1, 32>>>(A, B, D);
    cudaDeviceSynchronize();
    printf("hello_nixnan done: d=%g,%g f=%g,%g,%g D[0]=%g\n", d[0], d[1], f[0], f[1], f[2], D[0]);
    return 0;
}
