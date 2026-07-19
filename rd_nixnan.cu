// rd_nixnan.cu
// -----------------------------------------------------------------------------
// Reaction-diffusion FTCS demo for the nixnan tutorial.
//
//     du/dt = D u_xx + lambda u,   x in [0,L],   u(0,t)=u(L,t)=0
//     u(x,0) = sin(pi x / L)          (fundamental mode, satisfies the BCs)
//
// Explicit forward-Euler + central difference:
//     u_i^{n+1} = u_i + r (u_{i+1} - 2 u_i + u_{i-1}) + (lambda dt) u_i
//     r = D dt / dx^2
//
// The reaction term makes the mode grow like exp((lambda - D pi^2) t), so the
// grid values march straight up the FP exponent axis until they hit each
// format's overflow edge.  We run the SAME step in three precisions, one kernel
// each, so nixnan attributes exceptions and the exponent histogram per format:
//
//   FP16 (5-bit exp, max 65504) : overflows ~step 227 (t~0.23); diffusion
//                                 increments go SUBNORMAL near the boundaries.
//   BF16 (8-bit exp, ~3.4e38)   : overflows ~step 1818 (t~1.82); 8-bit mantissa
//                                 makes the u_{i+1}-2u_i+u_{i-1} cancellation
//                                 pure noise (precision loss, not an exception).
//   FP32                        : overflows ~step 1818, clean mantissa.
//
// Build:  nvcc -arch=sm_86 -lineinfo rd_nixnan.cu -o rd_nixnan
// Run  :  ./rd_nixnan                                   # baseline blow-up
//         LD_PRELOAD=/path/nixnan.so ./rd_nixnan        # exception summary
//         HISTOGRAM=1 LD_PRELOAD=/path/nixnan.so ./rd_nixnan
//         BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=/path/nixnan.so ./rd_nixnan
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef N
#define N 101          // grid points; dx = L/(N-1) = 0.01
#endif
#ifndef M
#define M 2500         // time steps; dt = 1e-3 -> T = 2.5
#endif

// --- physical / numerical constants (host + device) ---
static const double L      = 1.0;
static const double D      = 0.01;
static const double LAMBDA = 50.0;
static const double DX     = L / (N - 1);      // 0.01
static const double DT     = 1e-3;
static const double R      = D * DT / (DX * DX);   // 0.1   (diffusion number)
static const double LDT    = LAMBDA * DT;          // 0.05  (reaction per step)

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){                \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__);\
    exit(1);} } while(0)

// =============================================================================
// One explicit FTCS step, three precisions, three kernels.
// Everything stays IN-TYPE so nixnan sees genuine HADD/HMUL (f16), the bf16
// variants, and FADD/FFMA (f32) -- no silent promotion to float.
// =============================================================================

__global__ void rd_step_fp16(const __half* u, __half* un, __half r, __half lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = __float2half(0.f); return; }   // Dirichlet

    __half two  = __float2half(2.f);
    __half lap  = __hsub(__hadd(u[i + 1], u[i - 1]), __hmul(two, u[i])); // 2nd diff
    __half diff = __hmul(r, lap);
    __half reac = __hmul(lam, u[i]);
    un[i] = __hadd(__hadd(u[i], diff), reac);
}

__global__ void rd_step_bf16(const __nv_bfloat16* u, __nv_bfloat16* un,
                             __nv_bfloat16 r, __nv_bfloat16 lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = __float2bfloat16(0.f); return; }

    __nv_bfloat16 two  = __float2bfloat16(2.f);
    __nv_bfloat16 lap  = __hsub(__hadd(u[i + 1], u[i - 1]), __hmul(two, u[i]));
    __nv_bfloat16 diff = __hmul(r, lap);
    __nv_bfloat16 reac = __hmul(lam, u[i]);
    un[i] = __hadd(__hadd(u[i], diff), reac);
}

__global__ void rd_step_fp32(const float* u, float* un, float r, float lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = 0.f; return; }

    float lap = (u[i + 1] + u[i - 1]) - 2.f * u[i];
    un[i] = u[i] + r * lap + lam * u[i];   // FADD / FFMA in SASS
}

// =============================================================================
// Host drivers.  One per format so each blow-up timeline stays clean.
// Host-side conversions (__*2float / __float2*) are host-callable in CUDA 12.
// =============================================================================

template <class T> static T   to_T(double);              // convert double -> T
template <class T> static float from_T(T);               // convert T -> float
template <> __half           to_T<__half>(double v){ return __float2half((float)v); }
template <> __nv_bfloat16    to_T<__nv_bfloat16>(double v){ return __float2bfloat16((float)v); }
template <> float            to_T<float>(double v){ return (float)v; }
template <> float from_T<__half>(__half v){ return __half2float(v); }
template <> float from_T<__nv_bfloat16>(__nv_bfloat16 v){ return __bfloat162float(v); }
template <> float from_T<float>(float v){ return v; }

template <class T, class Kernel>
static void run(const char* tag, Kernel kern)
{
    std::vector<T> h(N);
    for (int i = 0; i < N; ++i)
        h[i] = to_T<T>(std::sin(M_PI * (i * DX) / L));

    T *u, *un;
    CUDA_CHECK(cudaMalloc(&u,  N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&un, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(u, h.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    T r   = to_T<T>(R);
    T lam = to_T<T>(LDT);
    dim3 block(128), grid((N + 127) / 128);

    printf("\n===== %s : reaction-diffusion FTCS (N=%d, M=%d, dt=%.0e) =====\n",
           tag, N, M, DT);

    int first_bad = -1;
    for (int n = 0; n < M; ++n) {
        kern<<<grid, block>>>(u, un, r, lam);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u, un);

        if ((n + 1) % 100 == 0 || n == M - 1) {
            CUDA_CHECK(cudaMemcpy(h.data(), u, N * sizeof(T), cudaMemcpyDeviceToHost));
            float mx = -1e30f; bool bad = false;
            for (int i = 0; i < N; ++i) {
                float v = from_T<T>(h[i]);
                if (!std::isfinite(v)) bad = true;
                if (std::isfinite(v) && v > mx) mx = v;
            }
            if (bad && first_bad < 0) first_bad = n + 1;
            printf("  step %5d (t=%.3f)  max=%.4e%s\n",
                   n + 1, (n + 1) * DT, mx, bad ? "   <-- non-finite present" : "");
        }
    }
    if (first_bad > 0)
        printf("  first non-finite at step %d (t=%.3f)\n", first_bad, first_bad * DT);

    cudaFree(u); cudaFree(un);
}

int main()
{
    printf("r = %.4f (diffusion number, stable if <= 0.5)\n", R);
    printf("lambda*dt = %.4f (reaction growth per step ~ %.4f x)\n", LDT, 1.0 + LDT);

    run<__half>       ("FP16", rd_step_fp16);
    run<__nv_bfloat16>("BF16", rd_step_bf16);
    run<float>        ("FP32", rd_step_fp32);

    printf("\nDone. Re-run under LD_PRELOAD=nixnan.so with HISTOGRAM=1 "
           "and BIN_SPEC_FILE=spec.json.\n");
    return 0;
}
