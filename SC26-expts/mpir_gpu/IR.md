# IR.md — Mixed-Precision IR Illustration: FP16 vs BF16(SIMT) vs BF16(Tensor Core)

Experimental plan for the proposal illustration. Algorithm 1.1 of Carson & Higham,
SISC 40(2):A817–A847, 2018 — stated there for an ARBITRARY correction solver
(assumptions (2.3)–(2.5)); we instantiate it with a one-time direct factorization
at u_f (Cholesky in the code; the proposal text says only "direct solver at u_f"
and names no factorization). Metric: iterations to fixed forward-error target,
as the performance factor.

## 1. Claim being illustrated

The scalar-u_f convergence theory (kappa_inf(A) <~ 1/u_f) ranks BF16 far below
FP16 (1/u_bf16 = 256 vs 1/u_fp16 = 2048). Hardware inverts part of this ranking:
tensor-core MMA accumulates BF16 products in FP32, so the *effective* solve
precision is finer than the nominal u_bf16. Consequence: a conditioning regime
exists (C2) where BF16-SIMT iterative refinement fails but BF16-TC converges.
Later phase: FP16's narrow exponent (10^+-5) produces overflow/Inf/NaN that
BF16 (10^+-38) is immune to — the axis the rounding-error model cannot see.

## 2. Precision configuration

Fixed triple (u_f, u, u_r) = (16-bit, FP32, FP64):

| symbol | role            | value                          |
|--------|-----------------|--------------------------------|
| u_f    | factorization   | FP16 (u=2^-11) or BF16 (u=2^-8)|
| u      | working         | FP32 (u=2^-24)                 |
| u_r    | residual        | FP64 (u=2^-53)                 |

Three factorization columns:

- **F16**  : FP16 storage rounding, FP32 accumulation (models FP16 SIMT-with-FMA / TC alike at this scale)
- **B16-S**: BF16 storage rounding, BF16 accumulation  (SIMT model)
- **B16-T**: BF16 storage rounding, FP32 accumulation  (tensor-core model)

## 3. Accuracy target (the "RE" for the proposal)

RE = ||x_hat - x_true||_inf / ||x_true||_inf <= 1e-6.

Rationale: limiting forward error of the triple is ~4p·u_r·cond(A,x) + u_fp32
~ 1e-7, so 1e-6 is attainable with margin; the originally proposed 1e-2 is hit
in 1–2 iterations by every convergent configuration and carries no signal.
Declare divergence if RE not met in maxit = 50, if error increases 3
consecutive iterations, or if the u_f factorization breaks down (see §9).
Report Niter and the median per-iteration contraction ratio rho = err_{i+1}/err_i.

## 4. Test matrices

Dense SPD, n = 256 (Colab): A = Q * diag(sigma) * Q^T, Q from QR of a Gaussian
matrix, sigma geometric from 1 down to 1/kappa_2 (mode-3 analogue of the paper's
gallery('randsvd'); SPD by construction so a pivoting-free direct solver
applies). x_true = N(0,1) vector, b = A x_true in FP64, rounded to FP32.
Fixed seed. 5 matrix instances per kappa; report median Niter.

## 5. Simulated arithmetic (Colab / NumPy — no native BF16, no TC)

- round16(X, fmt): FP16 via `astype(np.float16)`; BF16 via mantissa truncation of
  FP32 with round-to-nearest-even on the top 8 significand bits.
- Blocked right-looking Cholesky (A = R^T R), block size b = 32, no pivoting:
  - diagonal-block factor in FP32 on 16-bit-rounded data; round block factors to u_f;
  - trailing update C <- C - W W^T (SYRK-shaped, GEMM in the code):
    - **TC mode**   : product and accumulation in FP32; C stays FP32; factors rounded to u_f (error ~ c·u_f + O(n)·u_fp32);
    - **SIMT mode** : after each rank-b update, round C to u_f (error ~ (n/b)·u_f growth — proxy for per-FMA 16-bit accumulation).
- Correction solves (step 4): two triangular solves with R^T, R; SIMT mode
  rounds the intermediate solution every b entries to u_f; TC mode solves in
  FP32 on u_f-rounded factors.
- Residual: FP64 matvec, rounded to FP32. Update in FP32.
- Known model limitation, stated in the writeup: SIMT rounding at rank-b
  granularity under-counts true per-FMA accumulation error by <= factor b;
  Code2.cu removes the modeling entirely.

## 6. Calibration sweep, then freeze C1/C2/C3

The theory constant c·n·kappa·u_f has O(n) slack (paper: FP16 works to
kappa ~ 1e4, not 2048), so C values are frozen empirically:

Sweep kappa_2 in {1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5} x {F16, B16-S, B16-T};
record Niter/diverge/breakdown. Choose:

- **C1** : all three columns converge, Niter small and equal (control row).
- **C2** : F16 converges; B16-S diverges; B16-T converges. (C2a := B16-T@C2, C2b := B16-S@C2.)
- **C3** : all three fail (F16 included). Teaser for GMRES-IR / inner-solver
  upgrade (u_s decoupled from u_f) — the ATLAS running-example regime, where
  the same Algorithm 1.1 slot holds an AMG V-cycle instead of a factorization.

Provisional expectations (to be confirmed by the sweep): C1 ~ 1e1,
C2 ~ 3e2–1e3, C3 ~ 1e5.

## 7. Headline table (proposal figure)

| kappa_2 | F16 Niter | B16-S Niter | B16-T Niter |
|---------|-----------|-------------|-------------|
| C1      | .         | .           | .           |
| C2      | .         | DIVERGE     | .           |
| C3      | DIVERGE   | DIVERGE     | DIVERGE     |

Same RE = 1e-6 everywhere; Niter is the performance factor (per-iteration cost
fixed at O(n^2); O(n^3) factorization one-time and cheapest in the 16-bit
formats — the whole point).

## 8. Phases and deliverables

- **Phase A (this plan)** — IR.md. You review; on approval I execute Phase B.
- **Phase B — Code1.ipynb (Colab, NumPy)**: rounding utils, blocked Cholesky
  with TC/SIMT accumulation modes, IR driver, calibration sweep, frozen
  C1/C2/C3, headline table + convergence plots (err_inf vs iteration, log-y).
  I report the non-_gpu parameters: C1, C2, C3, n, b, RE, seeds, Niter table.
  You confirm on Colab.
- **Phase C — Code2.cu + Makefile (A100, sm_80)**: real arithmetic, no simulation.
  N_gpu ~ 4096–8192. Blocked Cholesky, no pivoting: TC path = cuBLAS GemmEx /
  SyrkEx with BF16 (CUDA_R_16BF) inputs and CUBLAS_COMPUTE_32F (TC FP32
  accumulate); SIMT path = hand-written __nv_bfloat16 GEMM kernel with bf16
  accumulation (no cuBLAS compute type accumulates in bf16 — this kernel IS the
  B16-S column); FP16 analogues via __half / CUBLAS_COMPUTE_16F. Triangular
  solves at u_f; residual via cublasDgemv. Instrument: print Niter per
  {C1_gpu, C2a_gpu, C2b_gpu, C3_gpu} at RE_GPU, plus per-iteration
  err/backward-error lines. Makefile: nvcc -arch=sm_80, targets all/run/clean.
  Parameters re-calibrated at N_gpu (thresholds shift with n).
- **Phase D — FP16 exception injection**: rescale A (and/or b) so entries push
  factor growth toward 65504: FP16 factorization overflows -> Inf/NaN propagate;
  BF16 columns unaffected at identical kappa. On GPU, observe with NixNan.
  Ties the illustration to the FP-exception-guard thread of the proposal.

## 9. Risks / honesty notes

- The TC-vs-SIMT gap appears only near the BF16 threshold (C2 row); C1 shows
  nothing — by design, say so in the caption.
- Rank-b rounding is a conservative SIMT model; hardware gap may be larger.
  Colab numbers are directional; A100 numbers are the citable ones.
- Low-precision Cholesky can break down (nonpositive computed pivot) once
  kappa·u_f ~ 1. Treated as a failure outcome in its own right — the threshold
  announcing itself — and logged separately from divergence. Standard remedy if
  needed for C2 tuning: Higham–Pranesh-style diagonal shift before rounding.
- Proposal text stays solver-generic: "direct factorization at u_f" as one
  instantiation of Algorithm 1.1's arbitrary correction solver; Cholesky is an
  implementation detail of the code, LU never mentioned.

## 10. Amendments after Phase B execution (2026-08-02)

- SIMT model tightened from rank-b to per-rank-1 trailing rounding (= exactly one
  rounding per FMA per element for the accumulation dimension); factorization is
  column (right-looking, rank-1) Cholesky; substitutions round running partials
  per column step. Block size b is thereby retired from the Colab model.
- Reference solution corrected: x_ref = FP64 solve of the STORED (A32, b32)
  system. Measuring against the unrounded system's x_true imposes a
  kappa*u_fp32 floor (~2e-5 at kappa=3e2) and falsely fails valid runs.
- Section-6 scaling (theta = ||r||_inf) implemented in the driver. Empirical
  finding: WITHOUT it, F16 stalls already at kappa=3e2 — residual entries fall
  below FP16 min normal 6.1e-5 and flush; BF16 unaffected. Logged as a Phase D
  preview (exponent axis bites before any forced overflow).
- Frozen parameters (n=256, RE=1e-6, seeds 1000-1004, median of 5):
  C1=1e2 (F16: 2 | B16-S: 12 | B16-T: 4), C2=1e3 (F16: 3 | B16-S: FAIL |
  B16-T: 14), C3=3e4 (F16: fail | B16-S/T: breakdown). Full sweep thresholds:
  F16 converges through 1e4 (Niter 2->11); B16-S through 3e2 (8->20);
  B16-T through 1e3 (3->14). TC accumulation buys ~4x in kappa and ~3-4x fewer
  iterations at every convergent kappa.

## 11. Amendment: F32 baseline column (2026-08-02)

Added F32 (u_f = u = FP32, no storage rounding, FP32 accumulation — the
"traditional" IR row). Sweep: Niter 0-1 at every kappa through 1e5 (threshold
1/u_fp32 ~ 1e8 far beyond the sweep). Role: positive control (C3 failures are
precision-caused) and the F32-vs-B16-T "interchangeable region" (C1-C2, same
delivered RE). Colab caveat: Niter alone always favors F32 — it prices nothing;
the trade appears in Phase C as total time T_fact(u_f) + Niter*T_iter, where
BF16-TC's ~16x cheaper factorization flops beat F32 inside the interchangeable
region and the crossover-vs-kappa curve is the proposal's precision-selection
payoff plot. Phase C option: TF32 as 5th config to isolate all three axes
(F16<->TF32 exponent, TF32<->F32 mantissa, B16-T<->B16-S accumulator).
Frozen headline with F32: C1 (1,2,12,4), C2 (1,3,FAIL,14), C3 (1,fail,bd,bd)
in column order (F32, F16, B16-S, B16-T).

## 12. Phase C/D specification (frozen with the GPU deliverable, 2026-08-02)

Deliverable: Code2.cu + Makefile + run_all.sh + README.md (this zip).
Five configs on A100: F32, TF32 (added: isolates mantissa axis vs F32,
exponent axis vs F16), F16, B16T (cuBLAS GemmEx COMPUTE_32F / FAST_TF32),
B16S (custom __hfma bf16-accumulate SYRK + single-block bf16 substitutions —
the only true bf16-arithmetic column, since no cuBLAS compute type
accumulates in bf16).

TIMING env var contract:
- TIMING=0: zero timing/scan calls inside the refinement loop; genuine loop
  time from a host clock outside it; t_per_iter = t_loop/steps. Factorization
  event-timed outside the loop in all modes.
- TIMING=1: loop-head-only instrumentation (one event pair + one non-finite
  scan of x, r); exactly one FIRST_EXCEPTION line (iter, ms since loop start)
  then stop; no other chatter.
- TIMING=2: chatter everywhere, timing statements confined to loop heads:
  per-iteration dt_ms/err/exc line + TOTAL line.

Measured per (config, kappa, seed): status, Niter, t_fact_ms, t_loop_ms,
t_per_iter_ms, first_exc_iter, first_exc_ms — RESULT line, parse-friendly.
Headline figure: total time (t_fact + t_loop) vs kappa at fixed RE_GPU=1e-6;
crossover of B16T vs F32 bounds the interchangeable region on hardware.

Phase D recipes (fp16 exceptions, timed to first arrival under TIMING=1):
--noscale --bscale 1e5 -> residual rounds to fp16 Inf (B16T contrast clean);
--noscale alone -> subnormal-flush stall (no Inf/NaN; TIMING=2 shows err
stagnation) reproducing the Colab sect-6 finding; --ascale for representation
overflow visible to NixNan in the first trailing GemmEx.

GPU parameters: N_gpu=4096 (8192 optional), NB=256, RE_GPU=1e-6, maxit=60,
seeds 0-4 (curand 1000+s). {C1,C2,C3}_gpu are frozen from `make sweep` ON
HARDWARE — the B16S threshold shifts down vs Colab (real per-FMA error grows
~sqrt(n)); provisional starting ladder 1e2/1e3/3e4.

Status: code authored and logic-matched to Code1.ipynb line by line, NOT
compiled here (no GPU in authoring environment); all API calls
macro-checked with file:line reporting. First-run errors to be sent back
for a fix pass.
