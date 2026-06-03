# cuTile-analysis - NixNan diagnostics for cu-tile kernels

This folder is the starting point for analyzing cu-tile based kernels with
[NixNan](https://github.com/parfloat/nixnan/).

The goal is to collect runnable kernels, captured NixNan traces, and short
notes that explain any observed floating-point exceptional values, exponent
range behavior, or other numerical diagnostics relevant to cu-tile workloads.

## Layout

```
cuTile-analysis/
  README.md
```

Future analyses can add one subdirectory per kernel, benchmark, or issue.
