# VA_ovflo_nixnan — NixNan findings

- **Sample:** `samples/quickstart/VA_ovflo_nixnan.py`
- **SAMPLING:** `0`  (0 = no sampling / instrument every launch; N = every Nth repeat of a kernel name)
- **Histogram bin threshold (COUNT):** `128`
- **exit_code:** `0`
- **raw `nixnan.nnlog` size:** 73,160 bytes

## TL;DR

**Expected positive control:** the sample is *designed* to overflow FP16 to `+Inf` in every lane. NixNan correctly catches 1 site with 1,023 repeats. Useful as a known-good signal for the rest of the sweep.

## NixNan exception / exponent-range report (tail of `nixnan.nnlog`)

```text
#nixnan: ------------ nixnan Report -----------

#nixnan: --- FP16 Operations ---
#nixnan: NaN:                    0 (0 repeats)
#nixnan: Infinity:               1 (1023 repeats)
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
#nixnan: Exponent range for f16: [zero, inf]
```
