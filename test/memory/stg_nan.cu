#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// @min_sm 80

// Memory path (STG). nixnan checks a stored value only when the register was last written by an
// instruction whose type it knows (src/meminstrumentation.cu: find_type), and it skips 8/16-bit
// stores (STG.E.U8 / STG.E.U16). So each kernel makes a NaN with an FP op (inf * 0), then stores it:
//   float   -> STG.E      (32-bit)  detected
//   double  -> STG.E.64   (64-bit)  detected
//   half2   -> STG.E      (packed)  detected
//   bf162   -> STG.E      (packed)  detected
//   half    -> STG.E.U16  (scalar)  NOT detected: documents a known gap

__global__ void k_f32(const float* in, float* out)  { out[threadIdx.x] = in[threadIdx.x] * 0.0f; }
__global__ void k_f64(const double* in, double* out){ out[threadIdx.x] = in[threadIdx.x] * 0.0; }
__global__ void k_h2(const half2* in, half2* out)   { out[threadIdx.x] = __hmul2(in[threadIdx.x], __float2half2_rn(0.0f)); }
__global__ void k_b2(const __nv_bfloat162* in, __nv_bfloat162* out) {
    out[threadIdx.x] = __hmul2(in[threadIdx.x], __float2bfloat162_rn(0.0f)); }
__global__ void k_h1(const half* in, half* out)     { out[threadIdx.x] = __hmul(in[threadIdx.x], __float2half(0.0f)); }

int main() {
    const int n = 32;
    float *f, *fo; double *d, *dd; half2 *h2, *h2o; __nv_bfloat162 *b2, *b2o; half *h, *ho;
    cudaMallocManaged(&f, n * sizeof(float));   cudaMallocManaged(&fo, n * sizeof(float));
    cudaMallocManaged(&d, n * sizeof(double));  cudaMallocManaged(&dd, n * sizeof(double));
    cudaMallocManaged(&h2, n * sizeof(half2));  cudaMallocManaged(&h2o, n * sizeof(half2));
    cudaMallocManaged(&b2, n * sizeof(__nv_bfloat162)); cudaMallocManaged(&b2o, n * sizeof(__nv_bfloat162));
    cudaMallocManaged(&h, n * sizeof(half));    cudaMallocManaged(&ho, n * sizeof(half));
    for (int i = 0; i < n; i++) {
        f[i] = INFINITY;
        d[i] = INFINITY;
        h2[i] = __float2half2_rn(INFINITY);
        b2[i] = __float2bfloat162_rn(INFINITY);
        h[i] = __float2half(INFINITY);
    }
    k_f32<<<1, n>>>(f, fo);
    k_f64<<<1, n>>>(d, dd);
    k_h2<<<1, n>>>(h2, h2o);
    k_b2<<<1, n>>>(b2, b2o);
    k_h1<<<1, n>>>(h, ho);
    cudaDeviceSynchronize();
    printf("stored: %g %g %g %g %g\n", fo[0], dd[0], __low2float(h2o[0]),
           __low2float(b2o[0]), __half2float(ho[0]));
    return 0;
}
