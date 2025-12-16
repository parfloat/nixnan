/**
 * testManyFMTs.cu - Exercise many binades of FP16, FP32, and FP64 values
 *
 * Generates 200 random numbers in range [0.0001, 16] for each format
 * and performs arithmetic operations to exercise floating-point units.
 *
 * Range spans roughly 2^-13 to 2^4 (~17 binades)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_VALUES 200
#define MIN_VAL 0.0001f
#define MAX_VAL 16.0f

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Generate random float in [min, max] using log-uniform distribution
// to cover binades more evenly
__device__ float rand_log_uniform(curandState* state, float min_val, float max_val) {
    float log_min = logf(min_val);
    float log_max = logf(max_val);
    float log_val = log_min + curand_uniform(state) * (log_max - log_min);
    return expf(log_val);
}

// Kernel to generate and exercise FP16 values
__global__ void exercise_fp16(half* values, half* results, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_VALUES) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate random value in range using log-uniform distribution
    float fval = rand_log_uniform(&state, MIN_VAL, MAX_VAL);
    half val = __float2half(fval);
    values[idx] = val;

    // Exercise: perform arithmetic operations
    half a = val;
    half b = __float2half(rand_log_uniform(&state, MIN_VAL, MAX_VAL));
    half c = __float2half(rand_log_uniform(&state, MIN_VAL, MAX_VAL));

    // Add, multiply, FMA operations
    half sum = __hadd(a, b);
    half prod = __hmul(a, b);
    half fma_result = __hfma(a, b, c);

    // More operations to exercise different binades
    half diff = __hsub(sum, prod);
    half final_result = __hadd(fma_result, diff);

    results[idx] = final_result;
}

// Kernel to generate and exercise FP32 values
__global__ void exercise_fp32(float* values, float* results, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_VALUES) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate random value in range using log-uniform distribution
    float val = rand_log_uniform(&state, MIN_VAL, MAX_VAL);
    values[idx] = val;

    // Exercise: perform arithmetic operations
    float a = val;
    float b = rand_log_uniform(&state, MIN_VAL, MAX_VAL);
    float c = rand_log_uniform(&state, MIN_VAL, MAX_VAL);

    // Add, multiply, FMA operations
    float sum = a + b;
    float prod = a * b;
    float fma_result = fmaf(a, b, c);

    // More operations to exercise different binades
    float diff = sum - prod;
    float final_result = fma_result + diff;

    results[idx] = final_result;
}

// Kernel to generate and exercise FP64 values
__global__ void exercise_fp64(double* values, double* results, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_VALUES) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate random value in range using log-uniform distribution
    double log_min = log(MIN_VAL);
    double log_max = log(MAX_VAL);
    double log_val = log_min + curand_uniform_double(&state) * (log_max - log_min);
    double val = exp(log_val);
    values[idx] = val;

    // Exercise: perform arithmetic operations
    double a = val;
    double b_log = log_min + curand_uniform_double(&state) * (log_max - log_min);
    double b = exp(b_log);
    double c_log = log_min + curand_uniform_double(&state) * (log_max - log_min);
    double c = exp(c_log);

    // Add, multiply, FMA operations
    double sum = a + b;
    double prod = a * b;
    double fma_result = fma(a, b, c);

    // More operations to exercise different binades
    double diff = sum - prod;
    double final_result = fma_result + diff;

    results[idx] = final_result;
}

// Helper to print binade distribution
void print_binade_distribution(const char* name, float* h_values, int n) {
    int binade_counts[32] = {0};  // binades from 2^-16 to 2^15

    for (int i = 0; i < n; i++) {
        float v = fabsf(h_values[i]);
        if (v == 0.0f) continue;
        int exp;
        frexpf(v, &exp);
        int binade_idx = exp + 16;  // offset to make positive
        if (binade_idx >= 0 && binade_idx < 32) {
            binade_counts[binade_idx]++;
        }
    }

    printf("%s binade distribution (2^exp):\n", name);
    for (int i = 0; i < 32; i++) {
        if (binade_counts[i] > 0) {
            printf("  2^%d: %d values\n", i - 16, binade_counts[i]);
        }
    }
    printf("\n");
}

void print_binade_distribution_double(const char* name, double* h_values, int n) {
    int binade_counts[32] = {0};

    for (int i = 0; i < n; i++) {
        double v = fabs(h_values[i]);
        if (v == 0.0) continue;
        int exp;
        frexp(v, &exp);
        int binade_idx = exp + 16;
        if (binade_idx >= 0 && binade_idx < 32) {
            binade_counts[binade_idx]++;
        }
    }

    printf("%s binade distribution (2^exp):\n", name);
    for (int i = 0; i < 32; i++) {
        if (binade_counts[i] > 0) {
            printf("  2^%d: %d values\n", i - 16, binade_counts[i]);
        }
    }
    printf("\n");
}

int main(int argc, char** argv) {
    printf("=== testManyFMTs: Exercising FP16, FP32, FP64 across binades ===\n");
    printf("Generating %d random values in range [%.4f, %.1f]\n", NUM_VALUES, MIN_VAL, MAX_VAL);
    printf("Range spans approximately 2^-13 to 2^4 (~17 binades)\n\n");

    unsigned long long seed = 12345ULL;
    if (argc > 1) {
        seed = strtoull(argv[1], NULL, 10);
    }
    printf("Using seed: %llu\n\n", seed);

    int blockSize = 256;
    int numBlocks = (NUM_VALUES + blockSize - 1) / blockSize;

    // ======================== FP16 ========================
    printf("--- FP16 (half precision) ---\n");
    half *d_fp16_values, *d_fp16_results;
    CHECK_CUDA(cudaMalloc(&d_fp16_values, NUM_VALUES * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_fp16_results, NUM_VALUES * sizeof(half)));

    exercise_fp16<<<numBlocks, blockSize>>>(d_fp16_values, d_fp16_results, seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and analyze
    half* h_fp16_values = (half*)malloc(NUM_VALUES * sizeof(half));
    half* h_fp16_results = (half*)malloc(NUM_VALUES * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_fp16_values, d_fp16_values, NUM_VALUES * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fp16_results, d_fp16_results, NUM_VALUES * sizeof(half), cudaMemcpyDeviceToHost));

    // Convert to float for analysis
    float* h_fp16_as_float = (float*)malloc(NUM_VALUES * sizeof(float));
    for (int i = 0; i < NUM_VALUES; i++) {
        h_fp16_as_float[i] = __half2float(h_fp16_values[i]);
    }
    print_binade_distribution("FP16 input", h_fp16_as_float, NUM_VALUES);

    printf("Sample FP16 values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", __half2float(h_fp16_values[i]));
    }
    printf("...\n");
    printf("Sample FP16 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", __half2float(h_fp16_results[i]));
    }
    printf("...\n\n");

    // ======================== FP32 ========================
    printf("--- FP32 (single precision) ---\n");
    float *d_fp32_values, *d_fp32_results;
    CHECK_CUDA(cudaMalloc(&d_fp32_values, NUM_VALUES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp32_results, NUM_VALUES * sizeof(float)));

    exercise_fp32<<<numBlocks, blockSize>>>(d_fp32_values, d_fp32_results, seed + 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float* h_fp32_values = (float*)malloc(NUM_VALUES * sizeof(float));
    float* h_fp32_results = (float*)malloc(NUM_VALUES * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_fp32_values, d_fp32_values, NUM_VALUES * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fp32_results, d_fp32_results, NUM_VALUES * sizeof(float), cudaMemcpyDeviceToHost));

    print_binade_distribution("FP32 input", h_fp32_values, NUM_VALUES);

    printf("Sample FP32 values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_fp32_values[i]);
    }
    printf("...\n");
    printf("Sample FP32 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_fp32_results[i]);
    }
    printf("...\n\n");

    // ======================== FP64 ========================
    printf("--- FP64 (double precision) ---\n");
    double *d_fp64_values, *d_fp64_results;
    CHECK_CUDA(cudaMalloc(&d_fp64_values, NUM_VALUES * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fp64_results, NUM_VALUES * sizeof(double)));

    exercise_fp64<<<numBlocks, blockSize>>>(d_fp64_values, d_fp64_results, seed + 2);
    CHECK_CUDA(cudaDeviceSynchronize());

    double* h_fp64_values = (double*)malloc(NUM_VALUES * sizeof(double));
    double* h_fp64_results = (double*)malloc(NUM_VALUES * sizeof(double));
    CHECK_CUDA(cudaMemcpy(h_fp64_values, d_fp64_values, NUM_VALUES * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fp64_results, d_fp64_results, NUM_VALUES * sizeof(double), cudaMemcpyDeviceToHost));

    print_binade_distribution_double("FP64 input", h_fp64_values, NUM_VALUES);

    printf("Sample FP64 values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.10f ", h_fp64_values[i]);
    }
    printf("...\n");
    printf("Sample FP64 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.10f ", h_fp64_results[i]);
    }
    printf("...\n\n");

    // ======================== Summary ========================
    printf("=== Summary ===\n");
    printf("Exercised %d values each for FP16, FP32, FP64\n", NUM_VALUES);
    printf("Operations performed per value: add, mul, fma, sub, add\n");
    printf("Total FP operations: %d\n", NUM_VALUES * 3 * 5);  // 3 formats * 5 ops each

    // Cleanup
    cudaFree(d_fp16_values);
    cudaFree(d_fp16_results);
    cudaFree(d_fp32_values);
    cudaFree(d_fp32_results);
    cudaFree(d_fp64_values);
    cudaFree(d_fp64_results);
    free(h_fp16_values);
    free(h_fp16_results);
    free(h_fp16_as_float);
    free(h_fp32_values);
    free(h_fp32_results);
    free(h_fp64_values);
    free(h_fp64_results);

    printf("\nDone!\n");
    return 0;
}
