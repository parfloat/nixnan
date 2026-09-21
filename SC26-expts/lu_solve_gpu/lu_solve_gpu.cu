// Self-contained CUDA (GPU) LU solver used as a nixnan instrumentation target.
//
// Same algorithm as the CPU lu_solve.cpp from the FPChecker tutorial
// (tutorial/example_1): LU factorization with partial pivoting (P A = L U)
// followed by forward/backward substitution, solving A x = 1. Matrices are
// flattened into row-major device buffers; the elimination step runs as CUDA
// kernels, and the two substitution passes run as single-thread kernels
// because of their sequential row-to-row dependency.
//
// On matrix.csv (well-conditioned) this produces a finite solution and a
// residual near 1e-16. On bad_matrix.csv (ill-conditioned) a near-zero pivot
// gets rounded to exactly 0.0 by the epsilon clamp in eliminate_kernel, and
// the resulting division by zero in backward_substitution_kernel produces
// Inf/NaN that propagates through the solution and residual -- this is the
// case nixnan is meant to catch.
//
// This file has no dependency on the tutorial repo: the CSV loader, matrix
// printer and residual helpers are inlined below instead of pulled from
// ../common, so it builds and runs standalone inside this experiment
// directory.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#define IDX(i, j, m) ((i) * (m) + (j))

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess)                                            \
        {                                                                     \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                 << cudaGetErrorString(err__) << endl;                       \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------
// Host helpers (inlined from tutorial/common/io.cpp and blas.cpp)
// -----------------------------------------------------------

static vector<vector<double>> load_matrix_from_csv(const string &filename)
{
    vector<vector<double>> matrix;
    ifstream file(filename);
    string line;

    if (!file.is_open())
    {
        cerr << "Error: Could not open file '" << filename << "'" << endl;
        return matrix;
    }

    while (getline(file, line))
    {
        vector<double> row;
        stringstream ss(line);
        string value;
        while (getline(ss, value, ','))
            row.push_back(stod(value));
        matrix.push_back(row);
    }

    return matrix;
}

static void print_matrix(const vector<vector<double>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (double v : row)
            cout << v << "\t";
        cout << endl;
    }
    cout << endl;
}

static vector<double> multiply_matrix_vector(const vector<vector<double>> &A, const vector<double> &x)
{
    int rows_A = A.size();
    int cols_A = rows_A ? A[0].size() : 0;
    vector<double> result(rows_A, 0.0);
    for (int i = 0; i < rows_A; ++i)
        for (int j = 0; j < cols_A; ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

static vector<double> subtract_vectors(const vector<double> &x, const vector<double> &y)
{
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        result[i] = x[i] - y[i];
    return result;
}

static double l2_norm(const vector<double> &v)
{
    double sum_of_squares = 0.0;
    for (double val : v)
        sum_of_squares += val * val;
    return sqrt(sum_of_squares);
}

// -----------------------------------------------------------
// Kernels
// -----------------------------------------------------------

__global__ void init_identity_kernel(double *M, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < m)
        M[IDX(i, j, m)] = (i == j) ? 1.0 : 0.0;
}

// Find the row (in [k, m)) with the largest |U[row][k]|.
__global__ void find_pivot_kernel(const double *U, int m, int k, int *pivot_row)
{
    extern __shared__ unsigned char smem[];
    double *smax = reinterpret_cast<double *>(smem);
    int *sidx = reinterpret_cast<int *>(smax + blockDim.x);

    int tid = threadIdx.x;
    int row = k + tid;

    double val = -1.0;
    int idx = k;
    if (row < m)
    {
        val = fabs(U[IDX(row, k, m)]);
        idx = row;
    }
    smax[tid] = val;
    sidx[tid] = idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && smax[tid + stride] > smax[tid])
        {
            smax[tid] = smax[tid + stride];
            sidx[tid] = sidx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        *pivot_row = sidx[0];
}

// Swap row k with row *pivot_row in U (all columns), P (all columns),
// and L (columns [0, k) only) -- same as apply_permutation bookkeeping
// in lu_factorization_partial_pivot() on the CPU side.
__global__ void swap_rows_kernel(double *U, double *L, double *P, int m, int k,
                                  const int *pivot_row_ptr)
{
    int pr = *pivot_row_ptr;
    if (pr == k)
        return;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m)
        return;

    double tmp = U[IDX(k, j, m)];
    U[IDX(k, j, m)] = U[IDX(pr, j, m)];
    U[IDX(pr, j, m)] = tmp;

    tmp = P[IDX(k, j, m)];
    P[IDX(k, j, m)] = P[IDX(pr, j, m)];
    P[IDX(pr, j, m)] = tmp;

    if (j < k)
    {
        tmp = L[IDX(k, j, m)];
        L[IDX(k, j, m)] = L[IDX(pr, j, m)];
        L[IDX(pr, j, m)] = tmp;
    }
}

// L[j][k] = U[j][k] / U[k][k] for j in (k, m).
__global__ void compute_multipliers_kernel(const double *U, double *L, int m, int k)
{
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m)
        L[IDX(j, k, m)] = U[IDX(j, k, m)] / U[IDX(k, k, m)];
}

// U[j][l] -= L[j][k] * U[k][l] for j in (k, m), l in [k, m), with the same
// "snap near-machine-epsilon values to zero" rule as the CPU version. This is
// the near-zero-pivot mechanism that eventually produces NaN on bad_matrix.csv.
__global__ void eliminate_kernel(double *U, const double *L, int m, int k)
{
    int j = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int l = k + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m && l < m)
    {
        double val = U[IDX(j, l, m)] - L[IDX(j, k, m)] * U[IDX(k, l, m)];
        if (fabs(val) < (10.0 * DBL_EPSILON))
            val = 0.0; // Set value to zero if close to machine epsilon
        U[IDX(j, l, m)] = val;
    }
}

// b' = P b
__global__ void apply_permutation_kernel(const double *P, const double *b, double *bp, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        double sum = 0.0;
        for (int j = 0; j < m; ++j)
            sum += P[IDX(i, j, m)] * b[j];
        bp[i] = sum;
    }
}

// Solve L y = b' (forward substitution). Sequential in i, so one thread.
__global__ void forward_substitution_kernel(const double *L, const double *bp, double *y, int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < i; ++j)
            sum += L[IDX(i, j, m)] * y[j];
        y[i] = bp[i] - sum; // L has 1s on the diagonal
    }
}

// Solve U x = y (backward substitution). Sequential in i, so one thread.
__global__ void backward_substitution_kernel(const double *U, const double *y, double *x, int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int i = m - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for (int j = i + 1; j < m; ++j)
            sum += U[IDX(i, j, m)] * x[j];
        x[i] = (y[i] - sum) / U[IDX(i, i, m)];
    }
}

// -----------------------------------------------------------
// Host driver
// -----------------------------------------------------------

static vector<double> flatten(const vector<vector<double>> &M)
{
    int m = M.size();
    vector<double> flat(m * m);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            flat[IDX(i, j, m)] = M[i][j];
    return flat;
}

static int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

static vector<double> solve_system_with_LU_gpu(const vector<vector<double>> &A,
                                                const vector<double> &b)
{
    int m = A.size();
    size_t mat_bytes = static_cast<size_t>(m) * m * sizeof(double);
    size_t vec_bytes = static_cast<size_t>(m) * sizeof(double);

    vector<double> h_A = flatten(A);

    double *d_U, *d_L, *d_P, *d_b, *d_bp, *d_y, *d_x;
    int *d_pivot_row;
    CUDA_CHECK(cudaMalloc(&d_U, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_P, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_bp, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_pivot_row, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_U, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vec_bytes, cudaMemcpyHostToDevice));

    dim3 idBlock(16, 16);
    dim3 idGrid((m + idBlock.x - 1) / idBlock.x, (m + idBlock.y - 1) / idBlock.y);
    init_identity_kernel<<<idGrid, idBlock>>>(d_L, m);
    init_identity_kernel<<<idGrid, idBlock>>>(d_P, m);
    CUDA_CHECK(cudaGetLastError());

    for (int k = 0; k < m - 1; ++k)
    {
        int pivotThreads = next_pow2(m - k);
        size_t shmem = pivotThreads * (sizeof(double) + sizeof(int));
        find_pivot_kernel<<<1, pivotThreads, shmem>>>(d_U, m, k, d_pivot_row);

        int swapThreads = 256;
        int swapBlocks = (m + swapThreads - 1) / swapThreads;
        swap_rows_kernel<<<swapBlocks, swapThreads>>>(d_U, d_L, d_P, m, k, d_pivot_row);

        int mulThreads = 256;
        int mulBlocks = (m - k - 1 + mulThreads - 1) / mulThreads;
        if (mulBlocks > 0)
            compute_multipliers_kernel<<<mulBlocks, mulThreads>>>(d_U, d_L, m, k);

        dim3 elimBlock(16, 16);
        dim3 elimGrid((m - k + elimBlock.x - 1) / elimBlock.x,
                       (m - k - 1 + elimBlock.y - 1) / elimBlock.y);
        if (elimGrid.x > 0 && elimGrid.y > 0)
            eliminate_kernel<<<elimGrid, elimBlock>>>(d_U, d_L, m, k);

        CUDA_CHECK(cudaGetLastError());
    }

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    apply_permutation_kernel<<<blocks, threads>>>(d_P, d_b, d_bp, m);
    forward_substitution_kernel<<<1, 1>>>(d_L, d_bp, d_y, m);
    backward_substitution_kernel<<<1, 1>>>(d_U, d_y, d_x, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> x(m);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, vec_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_U);
    cudaFree(d_L);
    cudaFree(d_P);
    cudaFree(d_b);
    cudaFree(d_bp);
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_pivot_row);

    return x;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <matrix file>" << endl;
        return 1;
    }

    string filename = argv[1];

    cout << "Loading matrix A:" << endl;
    vector<vector<double>> A = load_matrix_from_csv(filename);
    print_matrix(A);
    cout << "-----------------------------------------------------------" << endl;

    vector<double> b(A.size(), 1.0);
    auto x = solve_system_with_LU_gpu(A, b);

    cout << "Solution x:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        cout << "x[" << i << "]: " << x[i] << endl;
    cout << "-----------------------------------------------------------" << endl;

    auto a = multiply_matrix_vector(A, x);
    auto residual = subtract_vectors(a, b);
    auto norm = l2_norm(residual);
    cout << "Residual norm ||Ax - b||: " << norm << endl;

    return 0;
}
