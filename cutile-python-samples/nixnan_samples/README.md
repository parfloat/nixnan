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

`nixnan.nnlog` raw traces are committed for the small samples (FFT,
Transpose, MatMul, AllGatherMatmul, the two quickstarts, VA_ovflo).
For the heavy samples, only the gist (`findings.md`) plus `stdout.nnlog`
and `run.env` are committed — see `.gitignore` in this directory.

## Run profile

- `COUNT=128` — per-binade histogram bin threshold (matches the
  pytorch-issues sweep profile but at a coarser 128 grain).
- `HISTOGRAM=1`, `INSTR_MEM=1`, `PRINT_ILL_INSTR=1`, `ENABLE_FUN_DETAIL=0`,
  `LINE_INFO=0`, `RUN_CORRECTNESS=0`.
- `SAMPLING` is chosen per sample (heavy ⇒ larger N, light ⇒ 0 = no
  sampling). See `summary.tsv` and individual `run.env` for the exact
  value per sample.

| sample | SAMPLING | exit | raw nnlog | committed nnlog? |
|---|---:|---:|---:|---|
| [AllGatherMatmul](AllGatherMatmul/findings.md) | 10 | 0 | 92 B | yes |
| [AttentionFMHA](AttentionFMHA/findings.md) | 500 | 0 | 11.7 GB | **no** (gist only) |
| [BatchMatMul](BatchMatMul/findings.md) | 10 | 0 | 257 MB | **no** (gist only) |
| [FFT](FFT/findings.md) | 0 (full) | 0 | 101 KB | yes |
| [LayerNorm](LayerNorm/findings.md) | 5 | 0 | 734 MB | **no** (gist only) |
| [MatMul](MatMul/findings.md) | 10 | 0 | 103 MB | **no** (gist only) |
| [MoE](MoE/findings.md) | 10 | 0 | 1.9 GB | **no** (gist only) |
| [Transpose](Transpose/findings.md) | 5 | 0 | 26 MB | yes |
| [VectorAddition](VectorAddition/findings.md) | 5 | 0 | 119 MB | **no** (gist only) |
| [VectorAdd_quickstart](VectorAdd_quickstart/findings.md) | 0 | 0 | 143 KB | yes |
| [VA_ovflo_nixnan](VA_ovflo_nixnan/findings.md) | 0 | 0 | 73 KB | yes |

`SAMPLING=N>0` means: for each distinct kernel name, instrument every
Nth launch and skip the rest. `SAMPLING=0` instruments every launch.
FFT was deliberately run at SAMPLING=0 because it is the motivating
case.

## Sweep verdict

| Sample | NaN | +Inf | −Inf | Subnormal | Div/0 |
|---|---:|---:|---:|---:|---:|
| AllGatherMatmul | — | — | — | — | — *(empty trace; multi-process)* |
| **AttentionFMHA** | 0 | 0 | **972 (2,266,932)** FP32 | 752 (103,920) FP16 | 0 |
| BatchMatMul | 0 | 0 | 0 | 344 (3,912) FP16 | 0 |
| FFT | 0 | 0 | 0 | 102 (306) FP32 | **0** |
| LayerNorm | 0 | 0 | 0 | 0 | 0 |
| MatMul | 0 | 0 | 0 | 200 (1,240) FP16 | 0 |
| MoE | 0 | 0 | 0 | 0 | 0 |
| Transpose | 0 | 0 | 0 | 0 | 0 |
| VectorAddition | 0 | 0 | 0 | 3 (9) FP32 | 0 |
| VectorAdd_quickstart | 0 | 0 | 0 | 0 | 0 |
| VA_ovflo_nixnan | 0 | **1 (1,023)** FP16 *(designed)* | 0 | 0 | 0 |

Numbers are `distinct sites (total repeats)`.

### Headlines

1. **AttentionFMHA** is the only sample with a non-trivial new signal:
   972 FP32 `-Infinity` sites with ~2.27 M repeats. Likely the
   softmax mask sentinel (large negative additive scores becoming
   `-Inf` after `exp2` cancellation or accumulator underflow), but
   worth verifying that the `-Inf` is *designed* sentinel behavior
   and not arithmetic underflow.
2. **FFT** did not surface a divide-by-zero in this configuration
   (`SAMPLING=0`, no `--correctness-check`). The previously reported
   div-by-zero may live behind the correctness path or a different
   input config; rerun with `RUN_CORRECTNESS=1` to chase it.
3. **VA_ovflo_nixnan** is the expected positive control — confirms
   the toolchain is correctly detecting `+Inf` in FP16.
4. Subnormal counts on the FP16 matmul samples (BatchMatMul, MatMul,
   AttentionFMHA) are background noise from FP16 accumulation tails.
5. No NaN and no divide-by-zero anywhere outside the designed
   overflow control sample.
