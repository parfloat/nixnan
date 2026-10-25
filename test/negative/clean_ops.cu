#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

// @min_sm 80

// Negative test: ordinary values only, in every precision, plus one tensor-core multiply.
// nixnan must report zero in every section (no false positives).

__global__ void clean(const float* in, float* f, double* d, half* h, __nv_bfloat16* b) {
    int i = threadIdx.x;
    float x = in[i];                                   // 1.5 .. 2.5
    f[i] = x * 2.0f + 1.0f;                            // FFMA
    f[i] += sqrtf(x) + __expf(x * 0.1f) + 1.0f / x;    // MUFU.*
    d[i] = (double)x * 3.0 + 0.5;                      // DFMA / DMUL
    d[i] += 1.0 / (double)x;
    h[i] = __hfma(__float2half(x), __float2half(2.0f), __float2half(1.0f));   // HFMA2
    b[i] = __hfma(__float2bfloat16(x), __float2bfloat16(2.0f), __float2bfloat16(1.0f));
}

__global__ void tc(const half* a, const half* b, float* d) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(d, fc, 16, wmma::mem_row_major);
}

int main() {
    float *in, *f; double* d; half* h; __nv_bfloat16* b;
    cudaMallocManaged(&in, 32 * sizeof(float));
    cudaMallocManaged(&f, 32 * sizeof(float));
    cudaMallocManaged(&d, 32 * sizeof(double));
    cudaMallocManaged(&h, 32 * sizeof(half));
    cudaMallocManaged(&b, 32 * sizeof(__nv_bfloat16));
    for (int i = 0; i < 32; i++) in[i] = 1.5f + i / 32.0f;
    clean<<<1, 32>>>(in, f, d, h, b);

    half *A, *B; float* D;
    cudaMallocManaged(&A, 256 * sizeof(half));
    cudaMallocManaged(&B, 256 * sizeof(half));
    cudaMallocManaged(&D, 256 * sizeof(float));
    for (int i = 0; i < 256; i++) { A[i] = __float2half(0.5f); B[i] = __float2half(0.25f); }
    tc<<<1, 32>>>(A, B, D);
    cudaDeviceSynchronize();
    printf("f[0]=%g d[0]=%g h[0]=%g b[0]=%g D[0]=%g\n", f[0], d[0], __half2float(h[0]),
           __bfloat162float(b[0]), D[0]);
    return 0;
}
