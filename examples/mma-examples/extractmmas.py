import re

pattern = (
    r"fragment<[^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*([^,>]+)>"  # First fragment
    r".*?fragment<[^,]+,\s*\d+,\s*\d+,\s*\d+,\s*([^,>]+),"     # Second fragment
    r".*?fragment<[^,]+,\s*\d+,\s*\d+,\s*\d+,\s*([^,>]+),"     # Third fragment
    r".*?fragment<[^,]+,\s*\d+,\s*\d+,\s*\d+,\s*([^,>]+)>"     # Fourth fragment
)

mmas = set()

with open('/usr/include/crt/mma.h', 'r') as f:
    for line in f.readlines():
        matches = re.search(pattern, line)
        if matches:
            mmas.add(matches.groups())

print("""#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>

using namespace nvcuda::wmma::precision;
using namespace nvcuda::wmma;
using namespace nvcuda;""")

def getfname(M, N, K, dtype, atype, btype, ctype):
    return f"mma_{M}_{N}_{K}_{dtype}_{atype}_{btype}_{ctype}"

def get_kernel(t):
    M, N, K, dtype, atype, btype, ctype = t
    astore = "float" if atype == "tf32" else atype
    bstore = "float" if btype == "tf32" else btype
    fname = getfname(M, N, K, dtype, atype, btype, ctype)
    return """__global__ void {fname}({astore} *A, {bstore} *B, {ctype} *C, {dtype} *D)
{{
    wmma::fragment<wmma::matrix_a, {M}, {N}, {K}, {atype}, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, {M}, {N}, {K}, {btype}, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, {M}, {N}, {K}, {ctype}> c_frag;
    wmma::fragment<wmma::accumulator, {M}, {N}, {K}, {dtype}> d_frag;

    wmma::load_matrix_sync(a_frag, A, {K});
    wmma::load_matrix_sync(b_frag, B, {K});
    wmma::load_matrix_sync(c_frag, C, {N}, wmma::mem_row_major);
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(D, d_frag, {N}, wmma::mem_row_major);
}}""".format(fname=fname, astore=astore, bstore=bstore, M=M, N=N, K=K, dtype=dtype, atype=atype, btype=btype, ctype=ctype)

# Filter:
ns = set()
for m in mmas:
    inc = True
    for kw in ('char', 'int', 'u4', 'b1', 's4'):
        for x in m:
            if kw in x:
                inc = False
    if inc:
        ns.add(m)
mmas = {tuple(map(lambda x: x.replace('precision::', ''), n)) for n in ns}

for mma in mmas:
    print(get_kernel(mma))
    print()

print("""template <typename T>
T from_double(double d);

template<>
__half from_double<__half>(double d) {
    return __double2half(d);
}

template<>
nv_bfloat16 from_double<nv_bfloat16>(double d) {
    return __double2bfloat16(d);
}

template<>
float from_double<float>(double d) {
    return d;
}

template<>
double from_double<double>(double d) {
    return d;
}

template<typename T>
double to_double(T d);

template<>
double to_double<__half>(__half d) {
    return __half2float(d);
}

template<>
double to_double<nv_bfloat16>(nv_bfloat16 d) {
    return __bfloat162float(d);
}

template<>
double to_double<float>(float d) {
    return d;
}

template<>
double to_double<double>(double d) {
    return d;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
double mma_cpu_error(const Ta* a, const Tb* b, const Tc* c, Td* d, int M, int N, int K) {
    double error = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = to_double<Tc>(c[i*N + j]);
            for (int k = 0; k < K; k++) {
                sum += to_double<Ta>(a[i*K+k])*to_double<Tb>(b[j*K+k]);
            }
            // Assuming the computed value from the wmma is already in d
            double e = (to_double<Td>(d[i*N+j]) - sum);
            error += e > 0 ? e : -e;
            d[i*N + j] = from_double<Td>(sum);
        }
    }
    return error;
}


template <typename T>
void fill_mat(T* array, size_t rows, size_t cols, unsigned int seed) {
    // Initialize the random number generator with the fixed seed
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0f, 1.0); // Range [0.0, 1.0)

    // Fill the array in row-major order
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            array[i * cols + j] = from_double<T>(dist(rng));
        }
    }
}\n""")

def run_kernel(t):
    M, N, K, dtype, atype, btype, ctype = t
    astore = "float" if atype == "tf32" else atype
    bstore = "float" if btype == "tf32" else btype
    fname = getfname(M, N, K, dtype, atype, btype, ctype)
    return """{{
    {astore} *A;
    {bstore} *B;
    {ctype} *C;
    {dtype} *D;

    cudaMallocManaged((void **)&A, sizeof({astore}) * {M} * {K});
    cudaMallocManaged((void **)&B, sizeof({bstore}) * {K} * {N});
    cudaMallocManaged((void **)&C, sizeof({ctype}) * {M} * {N});
    cudaMallocManaged((void **)&D, sizeof({dtype}) * {M} * {N});
    fill_mat(A, {M}, {K}, 0);
    fill_mat(B, {K}, {N}, 1);
    fill_mat(C, {M}, {N}, 2);

    {fname}<<<1,32>>>(A, B, C, D);
    std::cout << "------- Running: {fname} -------\\n";
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {{
        std::cout << cudaGetErrorName(cuda_status) << std::endl;
        exit(1);
    }}
    for (int i = 0; i < {M}; i++) {{
        for (int j = 0; j < {N}; j++) {{
            std::cout << to_double<{dtype}>(D[{N}*i + j]) << ',';
        }}
        std::cout << std::endl;
    }}
    double error = mma_cpu_error<{astore},{bstore},{ctype},{dtype}>(A, B, C, D,{M}, {N}, {K});
    std::cout << "Total error: " << error << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
}}""".format(fname=fname, M=M, N=N, K=K, astore=astore, bstore=bstore, ctype=ctype, dtype=dtype)

print("int main() {")
for mma in mmas:
    print(run_kernel(mma))
print("return 0;\n}")