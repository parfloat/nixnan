# Mixed-Precision Cholesky-IR on A100 — Phase C/D (Code2.cu)

GPU realization of the Colab experiment (Code1.ipynb / IR.md). Carson & Higham
Algorithm 1.1; pivoting-free blocked Cholesky at u_f; u = FP32, u_r = FP64;
section-6 residual scaling on by default. All simulated roundings from the
Colab phase are replaced by real hardware arithmetic.

## Configurations

| config | factor storage | factorization accumulation | realized by |
|--------|---------------|---------------------------|-------------|
| F32    | fp32 | fp32 | cusolverDnSpotrf panels + cublasSsyrk trailing |
| TF32   | fp32 | TF32 tensor cores, fp32 accumulate | GemmEx CUBLAS_COMPUTE_32F_FAST_TF32 |
| F16    | fp16 | fp16 TC inputs, fp32 accumulate | GemmEx CUDA_R_16F / COMPUTE_32F |
| B16T   | bf16 | bf16 TC inputs, fp32 accumulate | GemmEx CUDA_R_16BF / COMPUTE_32F |
| B16S   | bf16 | bf16 SIMT, bf16 per-FMA accumulate | custom __hfma kernels (k_bf_syrk, k_bf_trsv) |

B16S is the only config whose trailing updates AND triangular solves run in
true bf16 arithmetic end to end (there is no cuBLAS compute type that
accumulates in bf16 — the custom kernels ARE that column). F16/B16T solves run
fp32 substitutions on u_f-rounded factors, with the residual rounded to u_f at
solve entry (Alg 1.1 step 3), matching the Colab TC model. TF32/F32 panels are
fp32; TF32 differs from F32 only in trailing-update compute type — i.e., its
u_f is "effective" in the accumulation-heavy O(n^3) part, which is the point.
Axis isolation: F16<->TF32 = exponent only, TF32<->F32 = mantissa only,
B16T<->B16S = accumulator only.

## Build / run

    make                 # nvcc -O3 -arch=sm_80 ... -lcublas -lcusolver -lcurand
    make run             # 5 configs x {1e2,1e3,3e4}, TIMING=0
    make sweep           # on-hardware calibration (freeze C*_gpu from this)
    make detail          # TIMING=2 per-iteration lines at kappa=1e3
    make phase_d         # fp16 exception runs (see below)
    ./run_all.sh         # full protocol -> sweep.log headline.log detail.log phased.log

CLI: `--n N --kappa K --config {F32|TF32|F16|B16T|B16S} --seed S --re RE
--maxit M --nb NB --sweep k1,k2,... --allconfigs --noscale --ascale S --bscale S`

Defaults: n=4096, kappa=1e3, re=1e-6, maxit=60, nb=256, seed=0 (curand seed
1000+seed, mirroring Colab's 1000-1004).

## TIMING env var (contract)

- `TIMING=0` — NO timing or scan calls inside the refinement loop; the loop
  runs untouched and its total wall time (host clock taken outside the loop)
  divided by executed steps gives a genuine `t_per_iter_ms`. Factorization is
  still event-timed (outside the loop; does not perturb it).
- `TIMING=1` — loop head only: one cudaEvent pair + one non-finite scan of
  x and r per iteration. Prints exactly one `FIRST_EXCEPTION` line at the
  first Inf/NaN (iteration index + ms since loop start), then stops that run
  (status=exception). No other chatter. Convergent runs print only RESULT.
- `TIMING=2` — chatter everywhere, but all timing statements remain at the
  loop head: one `iter= dt_ms= err= exc=` line per iteration, plus a TOTAL
  line. dt_ms is the time between consecutive loop heads (= one full
  residual+solve+update step).

Note the observer effect is confined by design: the loop body contains zero
timing/scan code in every mode; TIMING>=1 adds one event-sync and one O(n)
scan per iteration at the head. Compare t_per_iter_ms across TIMING=0 vs 2 to
quantify it (expect <0.1 ms).

## Output format

    RESULT config=B16T n=4096 kappa=1.00e+03 seed=0 status=converged niter=14 \
      t_fact_ms=... t_loop_ms=... t_per_iter_ms=... first_exc_iter=-1 first_exc_ms=-1.000

status in {converged, fail, breakdown, exception}. breakdown = nonpositive
Cholesky pivot at u_f (reported before any loop timing; niter=-1). The
proposal's total-time-vs-kappa crossover plot is t_fact_ms + t_loop_ms per
config per kappa at fixed RE.

## Phase D — forcing FP16 exceptions

- Overflow: `--noscale --bscale 1e5` amplifies b so the residual, rounded to
  fp16 at solve entry, exceeds 65504 -> Inf -> NaN in x. Run under TIMING=1 to
  get time-to-first-exception. The identical B16T run stays clean (bf16 range
  10^±38): the exponent axis, isolated.
- Underflow stall: `--noscale` alone. As convergence proceeds the residual
  drops below fp16 min normal 6.1e-5 and flushes to subnormals/zero; no
  Inf/NaN, so TIMING=1 stays silent — use TIMING=2 and watch err stagnate
  (status=fail). This reproduces the Colab finding that section-6 scaling is
  load-bearing for F16 at kappa >= 3e2.
- `--ascale S` scales A before storage for representation-overflow experiments
  (entries above 65504 become Inf already at the fp16 cast; NixNan should see
  them in the first trailing GemmEx).

## Expectations / calibration

Colab (n=256) frozen ladder: C1=1e2 (F32:1, F16:2, B16-S:12, B16-T:4),
C2=1e3 (1, 3, FAIL, 14), C3=3e4 (1, FAIL, breakdown, breakdown). At n=4096
the B16S threshold shifts DOWN (real per-FMA error grows ~sqrt(n)) and Niter
everywhere shifts up somewhat; run `make sweep` and freeze {C1,C2,C3}_gpu from
the transitions before quoting numbers. TF32 should track F32's Niter closely
while its t_fact approaches the 16-bit configs'.

Performance notes: B16S is deliberately slow — k_bf_syrk is a naive SIMT
kernel and k_bf_trsv is a single-block sequential solve; that column exists
for numerics (accumulator axis), and its t_fact also demonstrates why nobody
ships bf16-accumulate BLAS. F16/B16T t_fact is GemmEx-bound; expect the
fact-time ordering t(F32) > t(TF32) > t(B16T) ~ t(F16) at n >= 4096.

## Caveats

1. Authored against the Colab-validated reference logic but NOT compiled in
   the authoring environment (no GPU/nvcc available). Every API call is
   wrapped in CK/CB/CS/CR macros reporting file:line; if nvcc or first run
   complains, send the exact message back for a fix pass.
2. cusolverDnSpotrf workspace uses a fixed 8192-float scratch (c.tmp); for
   NB > 512 query sizes may exceed it — if Spotrf returns status 7
   (INVALID_WORKSPACE) raise the c.tmp allocation.
3. Matrix generation (QR of Gaussian, geometric spectrum) runs in fp64 on
   device; n=8192 needs ~1.6 GB for the fp64 copies — fine on 40/80 GB A100.
4. err is checked at every loop head against x_ref (fp64 solve of the stored
   fp32 system, as in Colab); this costs one axpy+amax per iteration in ALL
   modes. It is part of the algorithm's stopping test, not instrumentation,
   which is why it is allowed inside the TIMING=0 loop.
