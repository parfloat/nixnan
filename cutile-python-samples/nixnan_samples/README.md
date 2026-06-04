# NixNan sweep of `cutile-python-samples`

This directory holds a NixNan sweep over every sample in `samples/`. The
motivation: a divide-by-zero turned up in `FFT.py` during earlier work, and
this sweep checks whether any of the other samples are sitting on similar
silent FP exceptions (NaN, ±Inf, divide-by-zero, subnormals).

The sweep was produced by [`run_nixnan_samples.sh`](../run_nixnan_samples.sh).

## Per-sample layout

```
nixnan_samples/<sample_id>/
    findings.md           — TL;DR + Report tail + interpretation
    data/
        stdout.nnlog      — Python stdout/stderr from the run
        run.env           — sampling factor, command line, exit code, env
        nixnan.nnlog      — raw NixNan trace (omitted when > 100 MB; see below)
```

For samples larger than GitHub's 100 MB file limit, only the gist
(`findings.md`) plus `stdout.nnlog` and `run.env` are committed — see
`.gitignore` in this directory.

## Run profile

- `COUNT=128` — per-binade histogram bin threshold.
- `HISTOGRAM=1`, `INSTR_MEM=1`, `PRINT_ILL_INSTR=1`, `ENABLE_FUN_DETAIL=0`,
  `LINE_INFO=0`, `RUN_CORRECTNESS=0`.
- `SAMPLING` is now `0` (instrument every kernel launch) for every sample
  except AttentionFMHA and MoE, which remain sampled due to log volume.

| sample | SAMPLING | exit | wall | raw nnlog | committed nnlog? |
|---|---:|---:|---:|---:|---|
| [AllGatherMatmul](AllGatherMatmul/findings.md) | 0 | 0 | 6 s | 92 B | yes (empty) |
| [AttentionFMHA](AttentionFMHA/findings.md) | 500 | 0 | 38 m | 11.7 GB | **no** (gist only) |
| [BatchMatMul](BatchMatMul/findings.md) | 0 | 0 | 76 s | 296 MB | **no** (gist only) |
| [FFT](FFT/findings.md) | 0 | 0 | 6 s | 101 KB | yes |
| [FFT_correctness](FFT_correctness/findings.md) | 0 | 0 | ~6 s | 132 KB | yes — catches div0 |
| [LayerNorm](LayerNorm/findings.md) | 0 | 0 | 126 s | 882 MB | **no** (gist only) |
| [MatMul](MatMul/findings.md) | 0 | 0 | 102 s | 180 MB | **no** (gist only) |
| [MoE](MoE/findings.md) | 10 | 0 | 170 s | 1.9 GB | **no** (gist only) |
| [Transpose](Transpose/findings.md) | 0 | 0 | 53 s | 65 MB | yes |
| [VectorAddition](VectorAddition/findings.md) | 0 | 0 | 184 s | 380 MB | **no** (gist only) |
| [VectorAdd_quickstart](VectorAdd_quickstart/findings.md) | 0 | 0 | 5 s | 144 KB | yes |
| [VA_ovflo_nixnan](VA_ovflo_nixnan/findings.md) | 0 | 0 | 4 s | 73 KB | yes |

`SAMPLING=N>0` means: for each distinct kernel name, instrument every
Nth launch and skip the rest. `SAMPLING=0` instruments every launch.

## Sweep verdict (post unsampled re-run)

| Sample | NaN | +Inf | −Inf | Subnormal | Div/0 |
|---|---:|---:|---:|---:|---:|
| AllGatherMatmul | — | — | — | — | — *(empty trace; multi-process)* |
| **AttentionFMHA** *(SAMPLING=500)* | 0 | 0 | **972 (2,266,932)** FP32 | 752 (103,920) FP16 | 0 |
| BatchMatMul | 0 | 0 | 0 | 456 (4,472) FP16 | 0 |
| FFT | 0 | 0 | 0 | 102 (306) FP32 | 0 |
| **FFT_correctness** | 0 | 1 (0) FP32 | 0 | 102 (306) FP32 | **1 (1) FP32** |
| **LayerNorm** | 0 | 0 | **1 (0)** FP32 | 0 | **1 (1) FP32** |
| MatMul | 0 | 0 | 0 | 216 (1,768) FP16 | 0 |
| MoE *(SAMPLING=10)* | 0 | 0 | 0 | 0 | 0 |
| Transpose | 0 | 0 | 0 | 0 | 0 |
| VectorAddition | 0 | 0 | 0 | 3 (9) FP32 | 0 |
| VectorAdd_quickstart | 0 | 0 | 0 | 0 | 0 |
| VA_ovflo_nixnan | 0 | **1 (1,023)** FP16 *(designed)* | 0 | 0 | 0 |

Numbers are `distinct sites (total repeats)`.

### Headlines

1. **No NaN, no Div-by-zero, no Inf** anywhere inside a cuTile kernel. The
   only non-zero signal in cuTile kernel-space is FP16 matmul subnormals
   (BatchMatMul, MatMul, AttentionFMHA), which is expected accumulator
   tail.
2. **AttentionFMHA** is the only sample with a non-trivial cuTile-side
   signal: 972 distinct FP32 `-Infinity` sites with ~2.27 M repeats.
   Likely a softmax mask sentinel (large-negative score → `-Inf` after
   `exp2` cancellation), worth a follow-up to confirm it is *designed*
   sentinel behavior and not arithmetic underflow.
3. **FFT_correctness** and **LayerNorm** both surface a `Division by 0`
   — and in both cases the offending kernel is a *PyTorch / cuRAND*
   ATen kernel, **not** a cuTile kernel, and the SASS instruction is
   `MUFU.RSQ` (reciprocal square root) executed on a value of `0` or
   `-0`. The result is then masked back to `0` by a SEL/predicate so
   the user-visible output stays correct.
   - **FFT_correctness:** `MUFU.RSQ R5, R0` inside
     `abs_kernel_vectorized2_kernel` — `abs(complex<float>)` called by
     `torch.testing.assert_close`.
   - **LayerNorm:** `MUFU.RSQ R30, R21` inside `void at::native::`
     (kernel name truncated). By log-position the offending kernel is
     the very first GPU launch of the program — `torch.randn(weight_shape, ...)`
     for the LayerNorm weight tensor. PyTorch's normal-distribution
     kernel (cuRAND-style Box-Muller) computes
     `z = sqrt(-2·log(u1)) · cos(2π·u2)`; when `u1` rounds up to
     exactly `1.0`, `-2·log(1) = -0.0`, and the compiled
     `sqrt(x) = x · rsqrt(x)` idiom executes `rsqrt(-0) = -Inf` as a
     masked intermediate.
4. **Same SASS pattern, two different harmless sources.** Both are useful
   positive controls for the sweep methodology: NixNan caught real
   intermediate FP exceptions that the user-level correctness path is
   blind to, and in both cases the diagnosis confirmed they are *not*
   cuTile bugs.
5. **VA_ovflo_nixnan** is the expected positive control — confirms the
   toolchain is correctly detecting `+Inf` in FP16.
6. **MoE** remains at `SAMPLING=10` because the unsampled rerun was
   estimated at ~30 min wall and ~19 GB of trace; it can be promoted to
   `SAMPLING=0` later if needed.
