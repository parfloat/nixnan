# AllGatherMatmul — NixNan findings

- **Sample:** `samples/AllGatherMatmul.py`
- **SAMPLING:** `0`  (0 = no sampling / instrument every launch; N = every Nth repeat of a kernel name)
- **Histogram bin threshold (COUNT):** `128`
- **exit_code:** `0`
- **raw `nixnan.nnlog` size:** 92 bytes

## TL;DR

Empty NixNan report — the multi-process launcher spawns workers that each redirect their own `LOGFILE`; the parent log only captured load-time boilerplate. Re-run with `LOGFILE` set per-rank to inspect.

## NixNan exception / exponent-range report (tail of `nixnan.nnlog`)

```text
(no Report section emitted)
```
