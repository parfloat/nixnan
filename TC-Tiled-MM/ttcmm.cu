// From: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <cstdio>
#include <random>
#include <curand_kernel.h>
#include <iostream>

void cuda_check_error(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
        exit(1);
    }
}
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_Y = 0U, typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // The skew size could affect the data alignment in shared memory when we use vectorized load.
    // We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k)
        {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(
                &A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS -
                                                k};
            // Mask out the invalid elements.
            T* const A_row_vector_vals_ptr{
                reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y %
        // NUM_THREADS ==
        //               0U);
        if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
            A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        {
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_thread_block_tile_transposed
                    [A_thread_block_tile_col_idx + i]
                    [A_thread_block_tile_row_idx] =
                        reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n)
        {
            B_row_vector_vals = *reinterpret_cast<int4 const*>(
                &B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS -
                                                n};
            // Mask out the invalid elements.
            T* const B_row_vector_vals_ptr{
                reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // If this is true, the following if can be removed.
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %
        // NUM_THREADS ==
        //               0U);
        if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
            B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        {
            *reinterpret_cast<int4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) =
                B_row_vector_vals;
        }
    }
}

// GEMM kernel v07.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K]
                                               [BLOCK_TILE_SIZE_Y +
                                                BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X +
                                                        BLOCK_TILE_SKEW_SIZE_X];

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        c_frag;

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS, BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y>(
            A, lda, B, ldb, A_thread_block_tile_transposed, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
#pragma unrolls
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
        {
#pragma unroll
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
            {
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_tile_row_idx],
                    &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K]
                                                   [warp_row_idx *
                                                        WARP_TILE_SIZE_Y +
                                                    wmma_tile_row_idx *
                                                        WMMA_TILE_SIZE_Y],
                    BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
#pragma unroll
                for (size_t wmma_tile_col_idx{0U};
                     wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
                {
                    // These loads are extremely slow somehow, which affects the
                    // performance a lot. Load the fragment from shared memory.
                    nvcuda::wmma::load_matrix_sync(
                        b_frags[wmma_tile_col_idx],
                        &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K]
                                            [warp_col_idx * WARP_TILE_SIZE_X +
                                             wmma_tile_col_idx *
                                                 WMMA_TILE_SIZE_Y],
                        BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);

                    // Perform the matrix multiplication.
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                        a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // Load the fragment from shared memory.
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                n, nvcuda::wmma::mem_row_major);
            // Perform scaling and addition.
            for (size_t i{0}; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] =
                    alpha *
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] +
                    beta * c_frag.x[i];
            }
            // Store the fragment back to shared memory.
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void initialize_matrix_uniform(half* matrix, size_t rows, size_t cols,
    double a, double b, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed, idx, 0, &state);

    for (size_t i = idx; i < rows * cols; i += stride) {
        float random_val = curand_uniform(&state) * (b - a) + a;
        matrix[i] = __float2half(random_val);
    }
}

template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, const T* Corig, size_t ldc,
                                       cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    float elapsed_time_ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 50;

    for (int i = 0; i < N; ++i) {
        cudaMemcpy(C, Corig, m * n * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaEventRecord(start, stream);
        gemm_v07_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                            BLOCK_TILE_SIZE_K, BLOCK_TILE_SKEW_SIZE_X,
                            BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
                            WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y,
                            WMMA_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
            <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                                  *beta, C, ldc);
        cudaEventRecord(stop, stream);
        cuda_check_error(cudaDeviceSynchronize());
        cuda_check_error(cudaStreamSynchronize(stream));
        float iteration_time_ms = 0.0f;
        cudaEventElapsedTime(&iteration_time_ms, start, stop);
        elapsed_time_ms += iteration_time_ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Average execution time over %d runs: %f ms\n", N, elapsed_time_ms / N);
    printf("gflops: %f\n", (((float)m*(float) n)*((float)k+k-1))/elapsed_time_ms/1000000000.0);
}

int main() {
    size_t m = 2*(16384UL);
    size_t n = 2*(16384UL);
    size_t k = 2*(16384UL);
    half alpha = 1.0;
    half beta = 0.0;
    half *A, *B, *C, *Corig;
    cudaMalloc(&A, m * k * sizeof(half));
    cudaMalloc(&B, k * n * sizeof(half));
    cudaMalloc(&C, m * n * sizeof(half));
    cudaMalloc(&Corig, m * n * sizeof(half));

    initialize_matrix_uniform<<<128, 128, 0, 0>>>(A, m, k, 0.0, 1.0, 0);
    cuda_check_error(cudaDeviceSynchronize());
    initialize_matrix_uniform<<<128, 128, 0, 0>>>(B, k, n, 0.0, 1.0, 1);
    cuda_check_error(cudaDeviceSynchronize());
    initialize_matrix_uniform<<<128, 128, 0, 0>>>(Corig, m, n, 0.0, 1.0, 2);
    cuda_check_error(cudaDeviceSynchronize());

    launch_gemm_kernel_v07_vectorized(m, n, k, &alpha, A, m, B, k, &beta, C, Corig, m, 0);
    half *h_C = new half[m * n];
    cudaMemcpy(h_C, C, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        printf("%f\n", __half2float(h_C[i]));
    }
    delete[] h_C;
    cudaFree(Corig);
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);
    return 0;
}