// nvcc -o main main.cu -arch sm_86

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 1
#define N_TILES 1
#define K_TILES 1

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

using namespace nvcuda;

union hu {
	half f;
	unsigned short i;
};

half max_normal() {
	hu x;
	x.i = 0x7BFFU;
	return x.f;
}

half neg_max_normal() {
	hu x;
	x.i = 0x7BFFU | 0x8000U;
	return x.f;
}

half min_normal() {
	hu x;
	x.i = 0x0400U;
	return x.f;
}

half neg_min_normal() {
	hu x;
	x.i = 0x0400U | 0x8000U;
	return x.f;
}

half min_subnormal() {
	hu x;
	x.i = 0x0001U;
	return x.f;
}

half neg_min_subnormal() {
	hu x;
	x.i = 0x0001U | 0x8000U;
	return x.f;
}

__host__ void InitMatrix(half *A, half *B, half *C)
{
	for (size_t i = 0; i < 16; i++) {
		for (size_t j = 0; j < 16; j++) {
			size_t idx = i*16 + j;
			A[idx] = i == j ? 1.0 : 0.0;
			B[idx] = i == j ? 1.0 : 0.0;
			C[idx] = 0.0;
		}
	}
}

__global__ void WMMAF16TensorCore(half *A, half *B, half *C, half *D, half *AB)
{
	// Half precision warp matrix multiply and accumulate (WMMA)
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> ab_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> c_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> d_frag;
	
	wmma::fill_fragment(ab_frag, 0.0f);

	// AB = A*B + old(AB)
	wmma::load_matrix_sync(a_frag, A, M_TOTAL);
	wmma::load_matrix_sync(b_frag, B, K_TOTAL);
	wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
	// Save intermediate AB.
	wmma::store_matrix_sync(AB, ab_frag, M_TOTAL, wmma::mem_row_major);

	// D = A*B + C
	// This is the full warp matrix multiply and accumulate
    wmma::load_matrix_sync(c_frag, C, N_TOTAL, wmma::mem_row_major);
	wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

	// Store the output
	wmma::store_matrix_sync(D, d_frag, N_TOTAL, wmma::mem_row_major);
}

cudaError_t CalcWMMA(half *A, half *B, half *C, half *D, half* AB)
{
	cudaError_t cuda_status;
	dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 4 * WARP_SIZE; 
	blockDim.y = 4;

	gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
	gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

	// for Performance Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D, AB);
	cuda_status = cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// for Performance Metrics
	// printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
	// references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	// printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2) / milliseconds / 1e9);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return cuda_status;
}


void run_test(half A00, half B00, float C00, const char *name)
{
	cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		printf("cudaSetDevice failed! ");
		exit(1);
	}


	// Matrix on device
	half *A;
	half *B;
	half *AB;
	half *C;
	half *D;

	// CUDA Unified Memory 
	cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * K_TOTAL);
	cudaMallocManaged((void **)&B, sizeof(half) * K_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&AB, sizeof(half) * K_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&C, sizeof(half) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&D, sizeof(half) * M_TOTAL * N_TOTAL);
	
	// Init matrix A B C on host
	InitMatrix(A, B, C);

	A[0] = A00;
	B[0] = B00;
	C[0] = C00;

	printf("----- Test %s -----\n", name);
	printf("A[0,0] = %f (0x%04x), B[0,0] = %f (0x%04x), C[0,0] = %f (0x%04x)\n",
	       __half2float(A[0]), hu{.f=A[0]}.i, __half2float(B[0]), hu{.f=B[0]}.i, __half2float(C[0]), hu{.f=C[0]}.i);
	
	// computing gemm using tensor cores
	printf("Computing D = A * B + C with Tensor Cores...\n");

	// D = A * B + C, D holds the result after ret
	cuda_status = CalcWMMA(A, B, C, D, AB);

	printf("D[0,0]=%f (0x%04x)\n", __half2float(D[0]), hu {.f = D[0]}.i);

	// Print full matrix D
	// for (int i = 0; i < 16; i++) {
	// 	for (int j = 0; j < 16; j++) {
	// 		printf("%f, ", __half2float(D[i*16+j]));
	// 	}
	// 	printf("\n");
	// }
	
	cuda_status = cudaDeviceReset();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceReset failed! ");
		exit(1);
	}

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	cudaFree(AB);
	printf("--------------------------------------------------------------------------------\n");
}

int main() {
	run_test(max_normal(), max_normal(), 0.0, "overflow: A[0,0]=max_normal, B[0,0]=max_normal, C[0,0]=0");
	run_test(INFINITY, 1.0, -INFINITY, "NaN: A[0,0]=inf, B[0,0]=1, C[0,0]=-inf");

	run_test(min_normal(), 0.5, 0, "underflow mul: A[0,0]=min_normal, B[0,0]=0.5, C[0,0]=0.0");
	run_test(neg_min_normal(), 0.5, 0, "underflow mul: A[0,0]=neg_min_normal, B[0,0]=0.5, C[0,0]=0.0");

	run_test(min_subnormal(), 0.5, 0, "underflow mul: A[0,0]=min_subnormal, B[0,0]=0.5, C[0,0]=0.0");
	run_test(neg_min_subnormal(), 0.5, 0, "underflow mul: A[0,0]=neg_min_subnormal, B[0,0]=0.5, C[0,0]=0.0");
}