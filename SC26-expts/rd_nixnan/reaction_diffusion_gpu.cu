// CUDA port of the FPChecker tutorial's example_2/reaction_diffusion.cpp
// (SC26-Tutorial-NixNan/tutorial/example_2), used as a nixnan
// instrumentation target.
//
// Same PDE, same parameters, same algorithm as the CPU version:
//   1D linear reaction-diffusion equation
//     du/dt = D * d2u/dx2 + lambda * u
//   explicit finite-difference update, Dirichlet BCs u(0,t)=u(L,t)=0,
//   sine-pulse initial condition u(x,0) = sin(pi x / L).
//   L=1, T=4, N=101, D=0.01, lambda=25, M=80000 steps (dt = T/M = 5e-5).
//
// The tutorial's point: a large positive lambda amplifies the solution
// exponentially (~exp(lambda*dt*n)); by t=4 the growth factor is
// exp(lambda*T) = exp(100) =~ 2.7e43. That is still finite in FP64
// (max ~1.8e308) but exceeds FP32's range (max ~3.4e38), so the FP64
// build should run to completion while the FP32 build should overflow to
// Infinity, and then propagate NaN once Inf combines with itself in the
// diffusion stencil (Inf - 2*Inf + Inf = NaN).
//
// Precision is selected at compile time, exactly mirroring the CPU
// tutorial's "typedef double Real_t; // typedef float Real_t;" toggle:
// build with -DRD_USE_FLOAT for the FP32 variant, without it for the
// default FP64 variant (see Makefile: reaction_diffusion_gpu_fp64 vs
// reaction_diffusion_gpu_fp32).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#ifdef RD_USE_FLOAT
typedef float Real_t;
#else
typedef double Real_t;
#endif

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess)                                            \
        {                                                                     \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                       << ": " << cudaGetErrorString(err__) << std::endl;    \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

// Explicit FTCS update for interior points; boundaries pinned to 0
// (Dirichlet), exactly as in reaction_diffusion.cpp's time-stepping loop.
__global__ void rd_step_kernel(const Real_t *u, Real_t *u_next, int N,
                                Real_t D_dt_over_dx2, Real_t lambda_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    if (i == 0 || i == N - 1)
    {
        u_next[i] = (Real_t)0.0;
        return;
    }
    Real_t diffusion_term = D_dt_over_dx2 * (u[i + 1] - (Real_t)2.0 * u[i] + u[i - 1]);
    Real_t reaction_term = lambda_dt * u[i];
    u_next[i] = u[i] + diffusion_term + reaction_term;
}

// Single-block max reduction, deliberately as naive as std::max_element's
// pairwise comparisons (the CPU version's print_max_value uses
// std::max_element): once a NaN enters u, "v > best" is false for any
// comparison involving it, so a NaN never "wins" the reduction. This is
// the same silent-failure mode the CPU tutorial exhibits, not a bug
// introduced by the GPU port.
__global__ void max_reduce_kernel(const Real_t *u, int N, Real_t *out)
{
    extern __shared__ unsigned char smem_raw[];
    Real_t *sdata = reinterpret_cast<Real_t *>(smem_raw);
    int tid = threadIdx.x;

    Real_t best = u[0];
    for (int idx = tid; idx < N; idx += blockDim.x)
    {
        Real_t v = u[idx];
        if (v > best)
            best = v;
    }
    sdata[tid] = best;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && sdata[tid + stride] > sdata[tid])
            sdata[tid] = sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        *out = sdata[0];
}

static Real_t device_max(const Real_t *d_u, int N, Real_t *d_scratch, int threads)
{
    max_reduce_kernel<<<1, threads, threads * sizeof(Real_t)>>>(d_u, N, d_scratch);
    CUDA_CHECK(cudaGetLastError());
    Real_t h_max;
    CUDA_CHECK(cudaMemcpy(&h_max, d_scratch, sizeof(Real_t), cudaMemcpyDeviceToHost));
    return h_max;
}

static void print_max_value(const Real_t &max_val, double time)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "At time t = " << time << ", Maximum value of u = ";
    std::cout << std::scientific << std::setprecision(10) << (double)max_val << std::endl;
}

int main()
{
#ifdef RD_USE_FLOAT
    std::cout << "Precision: FP32" << std::endl;
#else
    std::cout << "Precision: FP64" << std::endl;
#endif

    // --- Parameters (identical to tutorial/example_2/reaction_diffusion.cpp) ---
    const Real_t L = 1.0;
    const Real_t T = 4.0;
    const int N = 101;
    const Real_t D = 0.01;
    const Real_t lambda = 25.0;
    const int M = 80000;
    const Real_t dt = T / M;
    const Real_t dx = L / (N - 1);

    Real_t stability_limit = (Real_t)1.0 / (2.0 * D / (dx * dx) + lambda);
    if (dt > stability_limit)
    {
        std::cerr << "WARNING: Chosen dt (" << (double)dt << ") might violate stability criterion ("
                   << (double)stability_limit << ") for explicit method." << std::endl;
    }
    else
    {
        std::cout << "Stability criterion satisfied: " << (double)dt << " <= "
                   << (double)stability_limit << std::endl;
    }

    std::cout << "Solving 1D Reaction-Diffusion PDE: du/dt = D*d2u/dx2 + lambda*u" << std::endl;
    std::cout << "Parameters: L=" << (double)L << ", T=" << (double)T << ", N=" << N
              << ", M=" << M << ", D=" << (double)D << ", lambda=" << (double)lambda << std::endl;
    std::cout << "Derived: dx=" << (double)dx << ", dt=" << (double)dt << std::endl;

    // --- Initial condition (host), copied to device ---
    std::vector<Real_t> h_u(N);
    for (int i = 0; i < N; ++i)
        h_u[i] = std::sin((Real_t)M_PI * (i * dx) / L);

    Real_t *d_u, *d_u_next, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_u, N * sizeof(Real_t)));
    CUDA_CHECK(cudaMalloc(&d_u_next, N * sizeof(Real_t)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(Real_t)));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), N * sizeof(Real_t), cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    print_max_value(device_max(d_u, N, d_scratch, threads), 0.0);

    const Real_t D_dt_over_dx2 = D * dt / (dx * dx);
    const Real_t lambda_dt = lambda * dt;
    const int print_interval = M / 50;

    double current_time = 0.0;
    bool reported_nonfinite = false;

    for (int n = 0; n < M; ++n)
    {
        current_time += (double)dt;

        rd_step_kernel<<<blocks, threads>>>(d_u, d_u_next, N, D_dt_over_dx2, lambda_dt);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_u, d_u_next);

        if ((n + 1) % print_interval == 0 || n == M - 1)
        {
            Real_t max_val = device_max(d_u, N, d_scratch, threads);
            print_max_value(max_val, current_time);
            if (!reported_nonfinite && !std::isfinite((double)max_val))
            {
                reported_nonfinite = true;
                std::cerr << "first non-finite reported max at step " << (n + 1)
                          << " (t=" << current_time << ")" << std::endl;
            }
        }
    }

    cudaFree(d_u);
    cudaFree(d_u_next);
    cudaFree(d_scratch);

    std::cout << "\nSimulation finished." << std::endl;
    return 0;
}
