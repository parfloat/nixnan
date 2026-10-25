#include <cstdio>
#include <cuda_bf16.h>

// @min_sm 80

// BF16 add / mul / fma, scalar and packed (bfloat162). On sm_80+ these compile to
// HFMA2.BF16_V2, the only BF16 arithmetic opcode nixnan instruments.
// Every current fp16/fp32/fp64 test has an all-zero BF16 section, so this is the BF16 coverage.

typedef __nv_bfloat16 bf;
typedef __nv_bfloat162 bf2;

__global__ void k_add(const bf* a, const bf* b, bf* d)  { int i = threadIdx.x; d[i] = __hadd(a[i], b[i]); }
__global__ void k_mul(const bf* a, const bf* b, bf* d)  { int i = threadIdx.x; d[i] = __hmul(a[i], b[i]); }
__global__ void k_fma(const bf* a, const bf* b, bf* d)  { int i = threadIdx.x; d[i] = __hfma(a[i], b[i], b[i]); }
__global__ void k_add2(const bf2* a, const bf2* b, bf2* d) { int i = threadIdx.x; d[i] = __hadd2(a[i], b[i]); }
__global__ void k_mul2(const bf2* a, const bf2* b, bf2* d) { int i = threadIdx.x; d[i] = __hmul2(a[i], b[i]); }
__global__ void k_fma2(const bf2* a, const bf2* b, bf2* d) { int i = threadIdx.x; d[i] = __hfma2(a[i], b[i], b[i]); }

static bf mk(unsigned short bits) { return __ushort_as_bfloat16(bits); }

int main() {
    const unsigned short MAX = 0x7F7F, MIN_NORMAL = 0x0080, HALF = 0x3F00, ONE = 0x3F80,
                         INF = 0x7F80, NINF = 0xFF80;
    // (a, b) pairs: overflow, inf + -inf (NaN), underflow to subnormal.
    const unsigned short pairs[][2] = {{MAX, MAX}, {INF, NINF}, {MIN_NORMAL, HALF}, {INF, ONE}};
    bf *A, *B, *D;
    cudaMallocManaged(&A, 64 * sizeof(bf));
    cudaMallocManaged(&B, 64 * sizeof(bf));
    cudaMallocManaged(&D, 64 * sizeof(bf));
    for (auto& p : pairs) {
        for (int i = 0; i < 64; i++) { A[i] = mk(p[0]); B[i] = mk(p[1]); }
        k_add<<<1, 32>>>(A, B, D);
        k_mul<<<1, 32>>>(A, B, D);
        k_fma<<<1, 32>>>(A, B, D);
        k_add2<<<1, 32>>>((bf2*)A, (bf2*)B, (bf2*)D);
        k_mul2<<<1, 32>>>((bf2*)A, (bf2*)B, (bf2*)D);
        k_fma2<<<1, 32>>>((bf2*)A, (bf2*)B, (bf2*)D);
        cudaDeviceSynchronize();
    }
    cudaFree(A); cudaFree(B); cudaFree(D);
    return 0;
}
