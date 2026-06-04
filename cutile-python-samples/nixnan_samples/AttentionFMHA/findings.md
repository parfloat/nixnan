# AttentionFMHA — NixNan findings

- **Sample:** `samples/AttentionFMHA.py`
- **SAMPLING:** `500`  (0 = no sampling / instrument every launch; N = every Nth repeat of a kernel name)
- **Histogram bin threshold (COUNT):** `128`
- **exit_code:** `0`
- **raw `nixnan.nnlog` size:** 11,696,100,665 bytes

## TL;DR

**Most signal of the sweep.** 972 FP32 `-Infinity` sites with **2,266,932 repeats** — consistent with the softmax mask pattern in flash-attention (positions outside the causal/key range receive a large negative additive score, which can show up as `-Inf` after `exp2` cancellation or accumulator underflow). 752 FP16 subnormals are background tile-edge noise from the tuned cuTile kernels (note the cudnn_generated SDPA tunings in the kernel list). No NaN or divide-by-zero observed. Worth opening as a follow-up: is the `-Inf` a *designed* mask sentinel (benign) or arithmetic underflow to `-Inf` (latent bug)?

## NixNan exception / exponent-range report (tail of `nixnan.nnlog`)

```text
#nixnan: ------------ nixnan Report -----------

#nixnan: --- FP16 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               0 (0 repeats)
#nixnan: -Infinity:              0 (0 repeats)
#nixnan: Subnormal:            752 (103920 repeats)
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
#nixnan: -Infinity:            972 (2266932 repeats)
#nixnan: Subnormal:              0 (0 repeats)
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
#nixnan: Exponent range for f16: [zero, 2]
#nixnan: Exponent range for f32: [zero, inf]
```

## Static-instruction warnings at load (first 8 unique)

```text
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R33, R29, R30, +INF  ;
#nixnan: #nixnan: warning: Infinite immediate found in operand 3 of @P1 FFMA R30, R20, R19, +INF  ;
```
