# FFT (with --correctness-check) — NixNan findings

- **Sample:** `samples/FFT.py --correctness-check`
- **SAMPLING:** `0` (instrument every kernel launch)
- **Histogram bin threshold (COUNT):** `128`
- **ENABLE_FUN_DETAIL:** `1`, **LINE_INFO:** `1`, **PRINT_ILL_INSTR:** `1`
- **exit_code:** `0` (Python `assert_close` reports "Correctness check passed")
- **raw `nixnan.nnlog` size:** 132,193 bytes

## TL;DR

**Re-running FFT with `--correctness-check` does surface the divide-by-zero**
that the earlier `SAMPLING=0` sweep missed. It is **not** in the cuTile
`fft_kernel` and **not** a true output corruption — it is a single
intermediate `MUFU.RSQ R5, R0` (reciprocal-square-root) executed with
`R0 == 0.0` inside PyTorch's `abs_kernel_vectorized2_kernel`, called by
`torch.testing.assert_close` during the correctness comparison.

Counts (FP32 only, all others zero):

| Class | distinct sites | repeats |
|---|---:|---:|
| Division by 0 | **1** | **1** |
| +Infinity | 1 | 0 |
| Subnormal | 102 | 306 |

The `Division by 0` and `+Infinity` events are the *same* MUFU.RSQ
instruction observed twice (once via operand-1 div0 detection, once via
operand-0 infinity-result detection) — see the diagnostic excerpt below.

## The offending kernel + instruction

From `nixnan.nnlog` (line numbers in the committed log):

```text
596: #nixnan: running kernel [abs_kernel_vectorized2_kernel] ...
   ...
630: #nixnan: running kernel [abs_kernel_vectorized2_kernel] ...
633: #nixnan: error [div0] detected in operand 1 of instruction MUFU.RSQ R5, R0 ; in function abs_kernel_vectorized2_kernel of type f32
634: #nixnan: error [infinity] detected in operand 0 of instruction MUFU.RSQ R5, R0 ; in function abs_kernel_vectorized2_kernel of type f32
```

- **Kernel:** `abs_kernel_vectorized2_kernel` — PyTorch's ATen vectorized
  CUDA kernel for `at::native::abs_kernel_cuda`, i.e. element-wise
  `torch.abs(x)`. For `complex<float>` inputs it computes
  `|z| = sqrt(re² + im²)`.
- **SASS instruction:** `MUFU.RSQ R5, R0` — the multi-function-unit
  reciprocal-square-root, `R5 := 1 / sqrt(R0)`.
- **Faulting operand:** `R0 == 0.0f` → `MUFU.RSQ(0)` produces `+Inf` →
  NixNan logs both events on the same instruction.
- **Why it's transient:** the compiled `abs(complex<float>)` uses the
  `x * rsqrt(x)` idiom for `sqrt(x)` (avoids the slower `MUFU.SQRT`).
  When `x == 0` the intermediate is `0 * Inf = NaN`, but the SASS guards
  the result with a select that returns `0.0` for the `x == 0` case.
  The final output is mathematically correct; only the MUFU.RSQ
  intermediate is undefined.

## Why didn't the earlier FFT sweep catch this?

The earlier run (`nixnan_samples/FFT/`) had `RUN_CORRECTNESS=0`, so
`assert_close` never ran and no `torch.abs(complex<float>)` was issued.
The cuTile `fft_kernel` itself contains no division-by-zero candidate
instruction (its twiddle / W matrices are pre-computed from
`torch.exp(-2πi · I·J / factor)` with `factor > 0`, so no host-side
divide hits zero either). NixNan correctly reported a clean trace for
the kernel-only run.

## Which element hits zero?

The two FFT-comparison `abs()` calls inside `assert_close` are
roughly:
```python
abs_diff      = (cutile_out - cufft_out).abs()      # |actual - expected|
abs_expected  = cufft_out.abs()                     # |expected|     (only for rtol)
```
With `N=8`, `BATCH_SIZE=2`, `torch.manual_seed(0)`, the cuTile output
matches cuFFT *exactly* in at least one bin (FFT bins are sums and
single-coefficient products — small `N` plus identical FP32 rounding
order produces exact bit-for-bit equality on at least one element).
That makes `abs(cutile_out - cufft_out)` = `0` for that element, and
the `MUFU.RSQ(0)` fires once. The single `(1 repeat)` count is
consistent with one zero-magnitude element in a 16-element tensor.

## Verdict

**Harmless intermediate exception inside PyTorch, not a cuTile FFT bug.**
The cuTile kernel is clean. The event is the well-known
`rsqrt(0)`-then-select pattern in `at::native::abs_kernel_cuda` for
zero-magnitude complex values.

This is a useful positive control for the sweep methodology: NixNan
caught a real (intermediate) FP exception that would have been invisible
to a pure correctness assertion (which still passes).

## NixNan exception / exponent-range report (tail of `nixnan.nnlog`)

```text
#nixnan: ------------ nixnan Report -----------

#nixnan: --- FP16 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               0 (0 repeats)
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:              0 (0 repeats)
#nixnan: Division by 0:          0 (0 repeats)

#nixnan: --- BF16 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               0 (0 repeats)
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:              0 (0 repeats)
#nixnan: Division by 0:          0 (0 repeats)

#nixnan: --- FP32 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               1 (0 repeats)
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:            102 (306 repeats)
#nixnan: Division by 0:          1 (1 repeats)

#nixnan: --- FP64 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               0 (0 repeats)
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:              0 (0 repeats)
#nixnan: Division by 0:          0 (0 repeats)

#nixnan: --- FP16 Memory  Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: --- BF16 Memory  Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: --- FP32 Memory  Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: --- FP64 Memory  Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: --- FP exponent ranges ---
#nixnan: Exponent range for f32: [zero, inf]
```
