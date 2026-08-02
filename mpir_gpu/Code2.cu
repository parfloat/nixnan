// Code2.cu — Mixed-precision Cholesky iterative refinement on A100 (sm_80).
// Phase C/D of IR.md. Carson & Higham SISC 2018, Algorithm 1.1; solver =
// pivoting-free blocked Cholesky at precision u_f; u = FP32, u_r = FP64.
//
// Configs (u_f storage | factorization accumulation):
//   F32  : fp32        | fp32 (cuSOLVER Spotrf + cublasSsyrk)         baseline
//   TF32 : fp32        | TF32 tensor cores, fp32 accumulate (GemmEx FAST_TF32)
//   F16  : fp16        | fp16 TC inputs, fp32 accumulate (GemmEx COMPUTE_32F)
//   B16T : bf16        | bf16 TC inputs, fp32 accumulate (GemmEx COMPUTE_32F)
//   B16S : bf16        | bf16 SIMT __hfma, bf16 accumulate (custom kernels)
//
// TIMING env var:
//   0 : no timing calls inside the refinement loop (genuine loop time);
//       only Niter + total loop wall time (host clock, outside loop) reported.
//   1 : loop-head cudaEvent + non-finite scan; prints ONE line at first
//       exception (iter index + ms since loop start); no other chatter.
//   2 : chatter everywhere, but timing statements only at loop heads:
//       one line per iteration (dt_ms, err, exception count).
//
// Build: make          Run: TIMING=0 ./ir_gpu --n 4096 --kappa 1e3 --config B16T
// NOTE: authored and logic-validated against the Colab reference
// (Code1.ipynb); not compiled in the authoring environment (no GPU) — every
// CUDA/cuBLAS/cuSOLVER call is wrapped in a checked macro, so any issue
// reports file:line on first run. Report errors back for a fix pass.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); exit(1);} } while(0)
#define CS(x) do { cusolverStatus_t s_=(x); if(s_!=CUSOLVER_STATUS_SUCCESS){ \
  fprintf(stderr,"cuSOLVER error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); exit(1);} } while(0)
#define CR(x) do { curandStatus_t s_=(x); if(s_!=CURAND_STATUS_SUCCESS){ \
  fprintf(stderr,"cuRAND error %d at %s:%d\n",(int)s_,__FILE__,__LINE__); exit(1);} } while(0)

enum Config { CFG_F32, CFG_TF32, CFG_F16, CFG_B16T, CFG_B16S, CFG_F64 };
static const char* CFG_NAME[] = {"F32","TF32","F16","B16T","B16S","F64"};

// ---------------------------------------------------------------- kernels ---
__global__ void k_d2s(const double* a, float* b, long m){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x; if(i<m) b[i]=(float)a[i]; }
__global__ void k_s2d(const float* a, double* b, long m){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x; if(i<m) b[i]=(double)a[i]; }
__global__ void k_scale_d(double* a, double s, long m){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x; if(i<m) a[i]*=s; }
// round-through u_f on an ld-strided submatrix (rows r0..r0+nr, cols c0..c0+nc)
__global__ void k_round16_sub(float* A, int ld, int r0, int c0, int nr, int nc, int isbf){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<nr && j<nc){
    float v = A[(r0+i) + (long)(c0+j)*ld];
    A[(r0+i) + (long)(c0+j)*ld] = isbf ? __bfloat162float(__float2bfloat16(v))
                                       : __half2float(__float2half(v));
  }
}
__global__ void k_round16_vec(float* x, int n, int isbf){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) x[i] = isbf ? __bfloat162float(__float2bfloat16(x[i]))
                      : __half2float(__float2half(x[i]));
}
// pack fp32 submatrix (ld) -> contiguous 16-bit (ld nr)
__global__ void k_pack_h(const float* A, int ld, int r0, int c0, int nr, int nc, __half* W){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<nr && j<nc) W[i + (long)j*nr] = __float2half(A[(r0+i) + (long)(c0+j)*ld]);
}
__global__ void k_pack_b(const float* A, int ld, int r0, int c0, int nr, int nc, __nv_bfloat16* W){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<nr && j<nc) W[i + (long)j*nr] = __float2bfloat16(A[(r0+i) + (long)(c0+j)*ld]);
}
__global__ void k_s2bf(const float* a, __nv_bfloat16* b, long m){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x; if(i<m) b[i]=__float2bfloat16(a[i]); }
__global__ void k_bf2s(const __nv_bfloat16* a, float* b, long m){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x; if(i<m) b[i]=__bfloat162float(a[i]); }
// cast bf16 panel (cols c0..c0+nc, rows r0..n) into fp32 buffer P (same global indexing, ld n)
__global__ void k_bfpanel2s(const __nv_bfloat16* A, float* P, int n, int r0, int c0, int nc){
  int i = blockIdx.x*blockDim.x + threadIdx.x;   // global row - r0
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(r0+i<n && j<nc) P[(r0+i) + (long)j*n] = __bfloat162float(A[(r0+i) + (long)(c0+j)*n]);
}
__global__ void k_spanel2bf(const float* P, __nv_bfloat16* A, int n, int r0, int c0, int nc){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(r0+i<n && j<nc) A[(r0+i) + (long)(c0+j)*n] = __float2bfloat16(P[(r0+i) + (long)j*n]);
}
// B16S trailing update: C[i,j] -= sum_t W[i,t]*W[j,t], one bf16 __hfma per t
// (per-FMA bf16 rounding = the SIMT accumulation model, now real arithmetic).
__global__ void k_bf_syrk(__nv_bfloat16* A, int n, int k0, int nb, int t0){
  int i = t0 + blockIdx.x*blockDim.x + threadIdx.x;   // global row  >= t0
  int j = t0 + blockIdx.y*blockDim.y + threadIdx.y;   // global col  >= t0
  if(i<n && j<n){
    __nv_bfloat16 acc = A[i + (long)j*n];
    for(int t=0;t<nb;t++){
      __nv_bfloat16 w1 = A[i + (long)(k0+t)*n];
      __nv_bfloat16 w2 = A[j + (long)(k0+t)*n];
      acc = __hfma(__hneg(w1), w2, acc);              // bf16 multiply-add, one rounding
    }
    A[i + (long)j*n] = acc;
  }
}
// B16S triangular solves, single block, bf16 per-op arithmetic.
// mode 0: forward  L y = r   ;  mode 1: backward L^T d = y
__global__ void k_bf_trsv(const __nv_bfloat16* L, __nv_bfloat16* y, int n, int mode){
  int tid = threadIdx.x, bs = blockDim.x;
  if(mode==0){
    for(int j=0;j<n;j++){
      if(tid==0) y[j] = __float2bfloat16(__bfloat162float(y[j]) / __bfloat162float(L[j+(long)j*n]));
      __syncthreads();
      __nv_bfloat16 yj = y[j];
      for(int i=j+1+tid;i<n;i+=bs) y[i] = __hfma(__hneg(L[i+(long)j*n]), yj, y[i]);
      __syncthreads();
    }
  } else {
    for(int j=n-1;j>=0;j--){
      if(tid==0) y[j] = __float2bfloat16(__bfloat162float(y[j]) / __bfloat162float(L[j+(long)j*n]));
      __syncthreads();
      __nv_bfloat16 yj = y[j];
      for(int i=tid;i<j;i+=bs) y[i] = __hfma(__hneg(L[j+(long)i*n]), yj, y[i]); // L^T[i,j]=L[j,i]
      __syncthreads();
    }
  }
}
// count non-finite entries (exception scan; active only when TIMING>=1)
__global__ void k_scan_nonfinite(const float* x, int n, int* cnt){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n && !isfinite(x[i])) atomicAdd(cnt,1);
}
// scale columns of Q by d (A = Q D Q^T construction)
__global__ void k_colscale(double* Q, const double* d, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<n && j<n) Q[i + (long)j*n] *= d[j];
}

static inline dim3 g1(long m){ return dim3((unsigned)((m+255)/256)); }
static inline dim3 g2(int a,int b){ return dim3((a+15)/16,(b+15)/16); }
static const dim3 B1(256), B2(16,16);

// ------------------------------------------------------------------ state ---
struct Ctx {
  int n; long nn;
  cublasHandle_t cb; cusolverDnHandle_t cs;
  double *A64, *Ac64, *b64, *x64, *r64, *xref64, *dtmp;
  float  *A32, *C, *Lf, *b32, *xref, *x, *r32, *rt, *d, *tmp;
  __nv_bfloat16 *Abf, *ybf;
  __half *Wh; __nv_bfloat16 *Wb; float *P;
  int *dinfo, *dcnt;
};

static float inf_norm(Ctx& c, const float* v, int n){
  int idx; CB(cublasIsamax(c.cb, n, v, 1, &idx));
  float h; CK(cudaMemcpy(&h, v+idx-1, sizeof(float), cudaMemcpyDeviceToHost));
  return fabsf(h);
}

// ---------------------------------------------------- problem generation ---
static void make_problem(Ctx& c, double kappa2, unsigned long seed, double ascale, double bscale){
  int n=c.n; long nn=c.nn;
  curandGenerator_t g; CR(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CR(curandSetPseudoRandomGeneratorSeed(g, seed));
  double *G=c.A64;                                     // reuse as scratch
  CR(curandGenerateNormalDouble(g, G, ((nn+1)/2)*2, 0.0, 1.0));
  // QR -> Q in place
  double* tau; CK(cudaMalloc(&tau, n*sizeof(double)));
  int lw1, lw2; CS(cusolverDnDgeqrf_bufferSize(c.cs, n, n, G, n, &lw1));
  CS(cusolverDnDorgqr_bufferSize(c.cs, n, n, n, G, n, tau, &lw2));
  int lw = lw1>lw2?lw1:lw2; double* work; CK(cudaMalloc(&work, (long)lw*sizeof(double)));
  CS(cusolverDnDgeqrf(c.cs, n, n, G, n, tau, work, lw, c.dinfo));
  CS(cusolverDnDorgqr(c.cs, n, n, n, G, n, tau, work, lw, c.dinfo));
  // D geometric: sigma_j = kappa^(-j/(n-1)) ; A = (Q D) Q^T
  double* dvec; CK(cudaMalloc(&dvec, n*sizeof(double)));
  { double* h=(double*)malloc(n*sizeof(double));
    for(int j=0;j<n;j++) h[j]=pow(kappa2, -(double)j/(double)(n-1));
    CK(cudaMemcpy(dvec,h,n*sizeof(double),cudaMemcpyHostToDevice)); free(h); }
  double* QD; CK(cudaMalloc(&QD, nn*sizeof(double)));
  CK(cudaMemcpy(QD, G, nn*sizeof(double), cudaMemcpyDeviceToDevice));
  k_colscale<<<g2(n,n),B2>>>(QD, dvec, n);
  double one=1.0, zero=0.0;
  CB(cublasDgemm(c.cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, QD, n, G, n, &zero, c.A64, n));
  if(ascale!=1.0) k_scale_d<<<g1(nn),B1>>>(c.A64, ascale, nn);
  // stored fp32 system, fp64 shadow of it
  k_d2s<<<g1(nn),B1>>>(c.A64, c.A32, nn);
  k_s2d<<<g1(nn),B1>>>(c.A32, c.Ac64, nn);
  // x_true, b64 = Ac64 * x_true, b32 = fl32(b64) (then optional bscale)
  CR(curandGenerateNormalDouble(g, c.x64, ((n+1)/2)*2, 0.0, 1.0));       // x64 <- x_true
  CB(cublasDgemv(c.cb, CUBLAS_OP_N, n, n, &one, c.Ac64, n, c.x64, 1, &zero, c.b64, 1));
  if(bscale!=1.0) k_scale_d<<<g1(n),B1>>>(c.b64, bscale, n);
  k_d2s<<<g1(n),B1>>>(c.b64, c.b32, n);
  k_s2d<<<g1(n),B1>>>(c.b32, c.b64, n);                                  // b64 := (double)b32
  // x_ref: fp64 solve of stored system
  CK(cudaMemcpy(c.dtmp, c.Ac64, nn*sizeof(double), cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(c.xref64, c.b64, n*sizeof(double), cudaMemcpyDeviceToDevice));
  int lwd; CS(cusolverDnDpotrf_bufferSize(c.cs, CUBLAS_FILL_MODE_LOWER, n, c.dtmp, n, &lwd));
  if(lwd>lw){ CK(cudaFree(work)); CK(cudaMalloc(&work,(long)lwd*sizeof(double))); lw=lwd; }
  CS(cusolverDnDpotrf(c.cs, CUBLAS_FILL_MODE_LOWER, n, c.dtmp, n, work, lw, c.dinfo));
  CS(cusolverDnDpotrs(c.cs, CUBLAS_FILL_MODE_LOWER, n, 1, c.dtmp, n, c.xref64, n, c.dinfo));
  k_d2s<<<g1(n),B1>>>(c.xref64, c.xref, n);
  CK(cudaFree(tau)); CK(cudaFree(work)); CK(cudaFree(dvec)); CK(cudaFree(QD));
  CR(curandDestroyGenerator(g));
  CK(cudaDeviceSynchronize());
}

// ------------------------------------------------------- factorizations ---
// returns 0 ok, >0 breakdown column (1-based-ish), and fills c.Lf (fp32
// round-through factors) for the cublas solve path; B16S fills c.Abf.
static int factorize(Ctx& c, Config cfg, int NB){
  int n=c.n; float one=1.f, mone=-1.f;
  if(cfg==CFG_F64){
    // F64: pure double precision (reference baseline)
    CK(cudaMemcpy(c.A64, c.Ac64, c.nn*sizeof(double), cudaMemcpyDeviceToDevice));
    for(int k=0;k<n;k+=NB){
      int nb=(k+NB<=n)? NB : n-k; int m=n-k-nb;
      int lwq; CS(cusolverDnDpotrf_bufferSize(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.A64 + k + (long)k*n, n, &lwq));
      CS(cusolverDnDpotrf(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.A64 + k + (long)k*n, n, c.dtmp, lwq, c.dinfo));
      int info; CK(cudaMemcpy(&info, c.dinfo, sizeof(int), cudaMemcpyDeviceToHost));
      if(info>0) return k+info;
      if(m>0){
        double moned=-1.0, oned=1.0;
        CB(cublasDtrsm(c.cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                       CUBLAS_DIAG_NON_UNIT, m, nb, &oned, c.A64 + k + (long)k*n, n,
                       c.A64 + (k+nb) + (long)k*n, n));
        CB(cublasDsyrk(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, m, nb, &moned,
                       c.A64 + (k+nb) + (long)k*n, n, &oned, c.A64 + (k+nb) + (long)(k+nb)*n, n));
      }
    }
    CK(cudaDeviceSynchronize());
    return 0;
  }
  if(cfg==CFG_B16S){
    k_s2bf<<<g1(c.nn),B1>>>(c.A32, c.Abf, c.nn);
    for(int k=0;k<n;k+=NB){
      int nb = (k+NB<=n)? NB : n-k; int m = n-k-nb;
      k_bfpanel2s<<<g2(n-k,nb),B2>>>(c.Abf, c.P, n, k, k, nb);
      int lwq; CS(cusolverDnSpotrf_bufferSize(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.P + k, n, &lwq));
      CS(cusolverDnSpotrf(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.P + k, n, c.tmp, lwq, c.dinfo));
      int info; CK(cudaMemcpy(&info, c.dinfo, sizeof(int), cudaMemcpyDeviceToHost));
      if(info>0) return k+info;
      if(m>0) CB(cublasStrsm(c.cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, m, nb, &one, c.P + k, n, c.P + k + nb, n));
      k_spanel2bf<<<g2(n-k,nb),B2>>>(c.P, c.Abf, n, k, k, nb);   // storage rounding to bf16
      if(m>0){
        dim3 g((m+15)/16,(m+15)/16);
        k_bf_syrk<<<g,B2>>>(c.Abf, n, k, nb, k+nb);
      }
    }
    CK(cudaDeviceSynchronize());
    return 0;
  }
  // fp32 working copy path (F32 / TF32 / F16 / B16T)
  CK(cudaMemcpy(c.C, c.A32, c.nn*sizeof(float), cudaMemcpyDeviceToDevice));
  int isbf = (cfg==CFG_B16T);
  if(cfg==CFG_F16 || cfg==CFG_B16T)
    k_round16_sub<<<g2(n,n),B2>>>(c.C, n, 0,0, n, n, isbf);      // representation rounding
  for(int k=0;k<n;k+=NB){
    int nb=(k+NB<=n)? NB : n-k; int m=n-k-nb;
    int lwq; CS(cusolverDnSpotrf_bufferSize(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.C + k + (long)k*n, n, &lwq));
    CS(cusolverDnSpotrf(c.cs, CUBLAS_FILL_MODE_LOWER, nb, c.C + k + (long)k*n, n, c.tmp, lwq, c.dinfo));
    int info; CK(cudaMemcpy(&info, c.dinfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(info>0) return k+info;
    if(cfg==CFG_F16 || cfg==CFG_B16T)
      k_round16_sub<<<g2(nb,nb),B2>>>(c.C, n, k, k, nb, nb, isbf);
    if(m>0){
      CB(cublasStrsm(c.cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                     CUBLAS_DIAG_NON_UNIT, m, nb, &one, c.C + k + (long)k*n, n,
                     c.C + (k+nb) + (long)k*n, n));
      if(cfg==CFG_F16 || cfg==CFG_B16T)
        k_round16_sub<<<g2(m,nb),B2>>>(c.C, n, k+nb, k, m, nb, isbf);
      if(cfg==CFG_F32){
        CB(cublasSsyrk(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, m, nb, &mone,
                       c.C + (k+nb) + (long)k*n, n, &one, c.C + (k+nb) + (long)(k+nb)*n, n));
      } else if(cfg==CFG_TF32){
        CB(cublasGemmEx(c.cb, CUBLAS_OP_N, CUBLAS_OP_T, m, m, nb, &mone,
                        c.C + (k+nb) + (long)k*n, CUDA_R_32F, n,
                        c.C + (k+nb) + (long)k*n, CUDA_R_32F, n, &one,
                        c.C + (k+nb) + (long)(k+nb)*n, CUDA_R_32F, n,
                        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
      } else if(cfg==CFG_F16){
        k_pack_h<<<g2(m,nb),B2>>>(c.C, n, k+nb, k, m, nb, c.Wh);
        CB(cublasGemmEx(c.cb, CUBLAS_OP_N, CUBLAS_OP_T, m, m, nb, &mone,
                        c.Wh, CUDA_R_16F, m, c.Wh, CUDA_R_16F, m, &one,
                        c.C + (k+nb) + (long)(k+nb)*n, CUDA_R_32F, n,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else { // B16T
        k_pack_b<<<g2(m,nb),B2>>>(c.C, n, k+nb, k, m, nb, c.Wb);
        CB(cublasGemmEx(c.cb, CUBLAS_OP_N, CUBLAS_OP_T, m, m, nb, &mone,
                        c.Wb, CUDA_R_16BF, m, c.Wb, CUDA_R_16BF, m, &one,
                        c.C + (k+nb) + (long)(k+nb)*n, CUDA_R_32F, n,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }
  }
  CK(cudaMemcpy(c.Lf, c.C, c.nn*sizeof(float), cudaMemcpyDeviceToDevice));
  CK(cudaDeviceSynchronize());
  return 0;
}

// correction solve: d <- (L L^T)^{-1} rin, at the config's solve precision.
static void corr_solve(Ctx& c, Config cfg, const float* rin, float* dout){
  int n=c.n;
  CK(cudaMemcpy(dout, rin, n*sizeof(float), cudaMemcpyDeviceToDevice));
  if(cfg==CFG_F16)  k_round16_vec<<<g1(n),B1>>>(dout, n, 0);   // Alg 1.1: round r to u_s
  if(cfg==CFG_B16T) k_round16_vec<<<g1(n),B1>>>(dout, n, 1);
  if(cfg==CFG_B16S){
    k_s2bf<<<g1(n),B1>>>(dout, c.ybf, n);
    k_bf_trsv<<<1,1024>>>(c.Abf, c.ybf, n, 0);
    k_bf_trsv<<<1,1024>>>(c.Abf, c.ybf, n, 1);
    k_bf2s<<<g1(n),B1>>>(c.ybf, dout, n);
  } else if(cfg==CFG_F64){
    // F64: solve in double precision (not used for correction in current flow)
    k_s2d<<<g1(n),B1>>>(rin, c.r64, n);
    CB(cublasDtrsv(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, c.A64, n, c.r64, 1));
    CB(cublasDtrsv(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, c.A64, n, c.r64, 1));
    k_d2s<<<g1(n),B1>>>(c.r64, dout, n);
  } else {
    CB(cublasStrsv(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, c.Lf, n, dout, 1));
    CB(cublasStrsv(c.cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, c.Lf, n, dout, 1));
  }
}

// ------------------------------------------------------------------ main ---
int main(int argc, char** argv){
  int n=4096, maxit=60, NB=256, seed=0, noscale=0;
  double kappa=1e3, RE=1e-6, ascale=1.0, bscale=1.0;
  const char* cfgname="B16T"; const char* sweep=nullptr; int allcfg=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--n")) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--kappa")) kappa=atof(argv[++i]);
    else if(!strcmp(argv[i],"--config")) cfgname=argv[++i];
    else if(!strcmp(argv[i],"--seed")) seed=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--re")) RE=atof(argv[++i]);
    else if(!strcmp(argv[i],"--maxit")) maxit=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--nb")) NB=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--noscale")) noscale=1;
    else if(!strcmp(argv[i],"--ascale")) ascale=atof(argv[++i]);
    else if(!strcmp(argv[i],"--bscale")) bscale=atof(argv[++i]);
    else if(!strcmp(argv[i],"--sweep")) sweep=argv[++i];
    else if(!strcmp(argv[i],"--allconfigs")) allcfg=1;
    else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
  }
  const char* te = getenv("TIMING"); int TIMING = te? atoi(te):0;

  Ctx c; c.n=n; c.nn=(long)n*n;
  CB(cublasCreate(&c.cb)); CS(cusolverDnCreate(&c.cs));
  CK(cudaMalloc(&c.A64,c.nn*sizeof(double))); CK(cudaMalloc(&c.Ac64,c.nn*sizeof(double)));
  CK(cudaMalloc(&c.dtmp,c.nn*sizeof(double)));
  CK(cudaMalloc(&c.b64,n*sizeof(double))); CK(cudaMalloc(&c.x64,n*sizeof(double)));
  CK(cudaMalloc(&c.r64,n*sizeof(double))); CK(cudaMalloc(&c.xref64,n*sizeof(double)));
  CK(cudaMalloc(&c.A32,c.nn*sizeof(float))); CK(cudaMalloc(&c.C,c.nn*sizeof(float)));
  CK(cudaMalloc(&c.Lf,c.nn*sizeof(float)));
  CK(cudaMalloc(&c.P,(long)n*NB>0? (long)n*NB*sizeof(float): sizeof(float)));
  CK(cudaMalloc(&c.b32,n*sizeof(float))); CK(cudaMalloc(&c.xref,n*sizeof(float)));
  CK(cudaMalloc(&c.x,n*sizeof(float))); CK(cudaMalloc(&c.r32,n*sizeof(float)));
  CK(cudaMalloc(&c.rt,n*sizeof(float))); CK(cudaMalloc(&c.d,n*sizeof(float)));
  CK(cudaMalloc(&c.tmp,c.nn>1024? 8192*sizeof(float): 8192*sizeof(float)));  // spotrf work
  CK(cudaMalloc(&c.Abf,c.nn*sizeof(__nv_bfloat16))); CK(cudaMalloc(&c.ybf,n*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&c.Wh,(long)n*NB*sizeof(__half))); CK(cudaMalloc(&c.Wb,(long)n*NB*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&c.dinfo,sizeof(int))); CK(cudaMalloc(&c.dcnt,sizeof(int)));

  double kappas[32]; int nk=0;
  if(sweep){ char* s=strdup(sweep); for(char* t=strtok(s,","); t && nk<32; t=strtok(NULL,",")) kappas[nk++]=atof(t); }
  else kappas[nk++]=kappa;
  Config cfgs[5]; int ncf=0;
  if(allcfg){ for(int i=0;i<6;i++) cfgs[ncf++]=(Config)i; }
  else { int f=-1; for(int i=0;i<6;i++) if(!strcmp(cfgname,CFG_NAME[i])) f=i;
         if(f<0){ fprintf(stderr,"bad --config\n"); return 1; } cfgs[ncf++]=(Config)f; }

  cudaEvent_t evF0,evF1,evH0,evH1,evL0;
  CK(cudaEventCreate(&evF0)); CK(cudaEventCreate(&evF1));
  CK(cudaEventCreate(&evH0)); CK(cudaEventCreate(&evH1)); CK(cudaEventCreate(&evL0));

  for(int ik=0; ik<nk; ik++){
    make_problem(c, kappas[ik], 1000+seed, ascale, bscale);
    float xref_norm = inf_norm(c, c.xref, n);
    for(int ic=0; ic<ncf; ic++){
      Config cfg = cfgs[ic];
      // ---- factorization (timed with events in all modes; outside the loop) --
      CK(cudaEventRecord(evF0));
      int bd = factorize(c, cfg, NB);
      CK(cudaEventRecord(evF1)); CK(cudaEventSynchronize(evF1));
      float t_fact; CK(cudaEventElapsedTime(&t_fact, evF0, evF1));
      if(bd>0){
        printf("RESULT config=%s n=%d kappa=%.2e seed=%d status=breakdown niter=-1 "
               "t_fact_ms=%.2f t_loop_ms=0 t_per_iter_ms=0 first_exc_iter=-1 first_exc_ms=-1\n",
               CFG_NAME[cfg], n, kappas[ik], seed, t_fact);
        continue;
      }
      // ---- initial solve (sect-6 scaling unless --noscale) -------------------
      float th0 = noscale? 1.0f : inf_norm(c, c.b32, n);
      CK(cudaMemcpy(c.rt, c.b32, n*sizeof(float), cudaMemcpyDeviceToDevice));
      float inv=1.0f/th0; CB(cublasSscal(c.cb, n, &inv, c.rt, 1));
      corr_solve(c, cfg, c.rt, c.x);
      CB(cublasSscal(c.cb, n, &th0, c.x, 1));
      // ---- refinement loop ---------------------------------------------------
      int niter=-1, bad=0, first_exc_iter=-1; float first_exc_ms=-1.f;
      const char* status="fail"; double prev_err=1e300;
      struct timespec w0,w1; clock_gettime(CLOCK_MONOTONIC,&w0);
      if(TIMING>=1){ CK(cudaEventRecord(evL0)); CK(cudaEventRecord(evH0)); }
      int steps=0; float dt_ms=0.f;
      for(int it=0; it<=maxit; ++it){
        // ---------------- loop head: ALL timing/scan lives here ---------------
        if(TIMING>=1){
          CK(cudaEventRecord(evH1)); CK(cudaEventSynchronize(evH1));
          CK(cudaEventElapsedTime(&dt_ms, evH0, evH1));
          CK(cudaEventRecord(evH0));
          int zero=0; CK(cudaMemcpy(c.dcnt,&zero,sizeof(int),cudaMemcpyHostToDevice));
          k_scan_nonfinite<<<g1(n),B1>>>(c.x, n, c.dcnt);
          k_scan_nonfinite<<<g1(n),B1>>>(c.r32, n, c.dcnt);
          int cnt; CK(cudaMemcpy(&cnt,c.dcnt,sizeof(int),cudaMemcpyDeviceToHost));
          if(cnt>0 && first_exc_iter<0){
            float t; CK(cudaEventElapsedTime(&t, evL0, evH1));
            first_exc_iter=it; first_exc_ms=t;
            printf("FIRST_EXCEPTION config=%s kappa=%.2e iter=%d t_ms=%.3f nonfinite=%d\n",
                   CFG_NAME[cfg], kappas[ik], it, t, cnt);
            if(TIMING==1){ status="exception"; break; }
          }
          if(TIMING==2 && it>0)
            printf("iter=%d dt_ms=%.3f err=%.3e exc=%d\n", it-1, dt_ms, prev_err, cnt);
        }
        // ---------------- body: untimed, unchattered ---------------------------
        // err
        CK(cudaMemcpy(c.tmp, c.x, n*sizeof(float), cudaMemcpyDeviceToDevice));
        float m1=-1.f; CB(cublasSaxpy(c.cb, n, &m1, c.xref, 1, c.tmp, 1));
        double err = (double)inf_norm(c, c.tmp, n) / (double)xref_norm;
        if(err<=RE){ status="converged"; niter=it; break; }
        if(it>0 && err>=prev_err){ if(++bad>=3){ status="fail"; break; } } else bad=0;
        prev_err=err;
        if(it==maxit){ status="fail"; break; }
        // residual at u_r
        k_s2d<<<g1(n),B1>>>(c.x, c.x64, n);
        CK(cudaMemcpy(c.r64, c.b64, n*sizeof(double), cudaMemcpyDeviceToDevice));
        double mo=-1.0, on=1.0;
        CB(cublasDgemv(c.cb, CUBLAS_OP_N, n, n, &mo, c.Ac64, n, c.x64, 1, &on, c.r64, 1));
        k_d2s<<<g1(n),B1>>>(c.r64, c.r32, n);
        // sect-6 scaling + correction + update at u
        float th = noscale? 1.0f : inf_norm(c, c.r32, n);
        if(th==0.f) th=1.f;
        CK(cudaMemcpy(c.rt, c.r32, n*sizeof(float), cudaMemcpyDeviceToDevice));
        float ith=1.0f/th; CB(cublasSscal(c.cb, n, &ith, c.rt, 1));
        corr_solve(c, cfg, c.rt, c.d);
        CB(cublasSaxpy(c.cb, n, &th, c.d, 1, c.x, 1));
        steps++;
      }
      CK(cudaDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC,&w1);
      double t_loop = (w1.tv_sec-w0.tv_sec)*1e3 + (w1.tv_nsec-w0.tv_nsec)*1e-6;
      double t_pi = steps>0? t_loop/steps : 0.0;
      printf("RESULT config=%s n=%d kappa=%.2e seed=%d status=%s niter=%d "
             "t_fact_ms=%.2f t_loop_ms=%.2f t_per_iter_ms=%.3f first_exc_iter=%d first_exc_ms=%.3f\n",
             CFG_NAME[cfg], n, kappas[ik], seed, status, niter, t_fact, t_loop, t_pi,
             first_exc_iter, first_exc_ms);
      if(TIMING==2)
        printf("TOTAL config=%s t_total_ms=%.2f (= t_fact + t_loop)\n", CFG_NAME[cfg], t_fact+t_loop);
    }
  }
  return 0;
}
