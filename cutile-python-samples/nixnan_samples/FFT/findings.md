# FFT — NixNan findings

- **Sample:** `samples/FFT.py`
- **SAMPLING:** `0`  (0 = no sampling / instrument every launch; N = every Nth repeat of a kernel name)
- **Histogram bin threshold (COUNT):** `128`
- **exit_code:** `0`
- **raw `nixnan.nnlog` size:** 101,297 bytes

## TL;DR

FP32 subnormals: 102 distinct sites (306 repeats). **No divide-by-zero observed in this run** (SAMPLING=0, full instrumentation, no `--correctness-check`). The static-instruction warnings at load show `FFMA R*, R*, R*, +INF` immediates — those are likely twiddle-factor rotation constants compiled with `+INF` placeholders, not runtime exceptions. If the previously observed div-by-zero needs to reappear, rerun with `--correctness-check` or a smaller `D`/atom-packing config.

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
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:            102 (306 repeats)
#nixnan: Division by 0:          0 (0 repeats)

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
#nixnan: Exponent range for f32: [zero, 31]
```

## Static-instruction warnings at load (first 8 unique)

```text
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R33, R29, R30, +INF  ;
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R30, R20, R19, +INF  ;
```
