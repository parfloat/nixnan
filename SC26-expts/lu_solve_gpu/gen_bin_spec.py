#!/usr/bin/env python3
"""Generate bin_spec.json for lu_solve_gpu's exp-binade (BIN_SPEC_FILE) run.

lu_solve_gpu.cu only computes in double precision (f64), so that is the only
format populated with exponent bins; f16/bf16/f32 stay empty per nixnan's
schema (see ../../Tutorial.md, "Binade-Targeted Range Monitoring").

Bucket size: each f64 bin spans 4 consecutive (unbiased) exponents, tiling
the full valid double exponent range [-1022, 1023] (subnormal/inf/nan
exponents excluded, matching the ranges used elsewhere in this repo, e.g.
SC26-expts/rd_nixnan/spec.json).

NOTE on "doublings": nixnan's docs (README.md, Tutorial.md) describe a
"doublings" field for adaptive threshold scaling, but as of this repo
checkout src/fp-histogram.cu never reads that key -- only "count" is used
(see the `count_threshold` report condition in src/fp-fun.cu: `record()`).
"doublings" is kept here for schema compatibility / forward-compatibility
with a future nixnan build, but it is currently a no-op.
"""

import json

LO, HI = -1022, 1023
BUCKET_SIZE = 4
COUNT_THRESHOLD = 128
DOUBLINGS = 7  # currently unused by nixnan; kept for schema compatibility


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
    f64_buckets = buckets(LO, HI, BUCKET_SIZE)
    spec_text = f"""{{
    "count": {COUNT_THRESHOLD},
    "doublings": {DOUBLINGS},
    "max_reports": 0,
    "f16": [],
    "bf16": [],
    "f32": [],
    "f64": {format_bucket_list(f64_buckets)}
}}
"""
    json.loads(spec_text)  # validate before writing
    with open("bin_spec.json", "w") as f:
        f.write(spec_text)
    print(f"Wrote bin_spec.json: {len(f64_buckets)} f64 buckets of width {BUCKET_SIZE}, "
          f"count threshold={COUNT_THRESHOLD}")


if __name__ == "__main__":
    main()
