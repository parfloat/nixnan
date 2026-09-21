#!/usr/bin/env python3
"""Generate bin_spec.json for reaction_diffusion_gpu's exp-binade
(BIN_SPEC_FILE) run.

Unlike lu_solve_gpu (which only ever computes in one precision per build),
reaction_diffusion_gpu.cu is built twice -- once for f64, once for f32 -- so
both formats are genuinely "present" across the two binaries that share
this one spec file. f16/bf16 stay empty; the CPU tutorial this is ported
from (example_2) only ever compares FP64 vs FP32.

Bucket size: each bin spans 4 consecutive (unbiased) exponents, tiling each
format's full valid range (subnormal/inf/nan exponent codes excluded):
  f64: [-1022, 1023]  (matches SC26-expts/lu_solve_gpu/gen_bin_spec.py)
  f32: [-126, 127]

NOTE on "doublings": as documented in
../lu_solve_gpu/README.md / lu_solve_gpu_report.tex, nixnan's
src/fp-histogram.cu does not currently read the "doublings" key -- only
"count" is used. It is kept here for schema compatibility only.
"""

import json

BUCKET_SIZE = 4
# reaction_diffusion_gpu runs M=80000 time steps over ~99 interior points
# per precision build -- roughly 1000x more instrumented FP-op instances
# than lu_solve_gpu's one-shot 20x20 solve. count=128 (lu_solve_gpu's
# value) produced ~950,000 report lines here; count=1,000,000 was chosen
# empirically to land back in the same "nice" ~100-150 reports/run range.
COUNT_THRESHOLD = 1_000_000
DOUBLINGS = 7  # currently unused by nixnan; kept for schema compatibility

F64_RANGE = (-1022, 1023)
F32_RANGE = (-126, 127)


def buckets(lo, hi, size):
    out = []
    x = lo
    while x <= hi:
        y = min(x + size - 1, hi)
        out.append([x, y])
        x = y + 1
    return out


def format_bucket_list(bucket_list, indent="        "):
    rows = ",\n".join(f"{indent}[{lo}, {hi}]" for lo, hi in bucket_list)
    return "[\n" + rows + "\n    ]"


def main():
    f64_buckets = buckets(*F64_RANGE, BUCKET_SIZE)
    f32_buckets = buckets(*F32_RANGE, BUCKET_SIZE)
    spec_text = f"""{{
    "count": {COUNT_THRESHOLD},
    "doublings": {DOUBLINGS},
    "max_reports": 0,
    "f16": [],
    "bf16": [],
    "f32": {format_bucket_list(f32_buckets)},
    "f64": {format_bucket_list(f64_buckets)}
}}
"""
    json.loads(spec_text)  # validate before writing
    with open("bin_spec.json", "w") as f:
        f.write(spec_text)
    print(f"Wrote bin_spec.json: {len(f32_buckets)} f32 buckets, "
          f"{len(f64_buckets)} f64 buckets, width {BUCKET_SIZE}, "
          f"count threshold={COUNT_THRESHOLD}")


if __name__ == "__main__":
    main()
