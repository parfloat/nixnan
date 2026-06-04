# LayerNorm — NixNan findings

- **Sample:** `samples/LayerNorm.py`
- **SAMPLING:** `0`  (0 = no sampling / instrument every launch; N = every Nth repeat of a kernel name)
- **Histogram bin threshold (COUNT):** `128`
- **exit_code:** `0`
- **raw `nixnan.nnlog` size:** 925,044,378 bytes

## TL;DR

**New signal at SAMPLING=0.** Exactly 1 FP32 `Division by 0` site (1 repeat) and 1 FP32 `-Infinity` site, both pointing at the same SASS instruction `MUFU.RSQ R30, R21` in a truncated `void at::native::` kernel. By log-position the offending kernel is the very first GPU kernel of the run (log line ~10 of ~1.7M), which corresponds to `torch.randn(weight_shape, ...)` at the top of the sample. PyTorch's normal-distribution kernel uses cuRAND-style Box-Muller (`sqrt(-2·log(u1)) · cos(2π·u2)`); when `u1` rounds to `1.0`, `-2·log(u1)` becomes `-0.0` and the compiled `sqrt(x) = x · rsqrt(x)` idiom executes `rsqrt(-0)` = `-Inf` as an intermediate (the result is then masked / never observed). Harmless intermediate exception, **not** a LayerNorm or cuTile bug — same pattern as the FFT `abs(complex<float>)` case in `FFT_correctness/`.

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
#nixnan: Infinity:               0 (0 repeats)
#nixnan: -Infinity:              1 (0 repeats)
#nixnan: Subnormal:              0 (0 repeats)
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
#nixnan: Exponent range for bf16: [zero, 2]
#nixnan: Exponent range for f32: [zero, inf]
```

## Static-instruction warnings at load (first 8 unique)

```text
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R33, R29, R30, +INF  ;
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R30, R20, R19, +INF  ;
```
