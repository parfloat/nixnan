# first_test — hello_nixnan

A 58-line CUDA program for demos. It raises floating-point exceptions in every precision, stores
NaNs to global memory (STG), and runs one tensor-core multiply that produces more.

    make -C .. first_test/hello_nixnan                  # or: nvcc -lineinfo -arch=native -o hello_nixnan hello_nixnan.cu
    LD_PRELOAD=../../nixnan.so ./hello_nixnan           # runs to completion, nixnan prints its report on exit

Needs sm_80+ (BF16 and the `HMMA.16816` tensor-core form).

| what | how | result |
|---|---|---|
| FP64 | `1.0 / 0.0`, `inf * 0` | +Inf and division by zero (`MUFU.RCP64H`), NaN (`DMUL`) |
| FP32 | `1.0f / 0.0f`, `inf * 0`, `FLT_MIN * 0.5` | Inf and div-by-zero (`MUFU.RCP`), NaN, subnormal (`FMUL`) |
| FP16 | `65504 * 65504`, then `* 0` (packed `half2`) | Inf, then NaN (`HMUL2`) |
| BF16 | `3e38 * 3e38`, then `* 0` (packed) | Inf, then NaN (`HFMA2.BF16_V2`) |
| memory | the NaN results above are written to global memory | NaN at `STG.E.64`, `STG.E` (FP32, FP16, BF16) |
| tensor core | 16x16x16 FP16 multiply, FP32 accumulate, `A[0,0] = Inf` | Inf inputs, NaN/Inf outputs (`HMMA.16816.F32`), NaN stored (`STG.E.64`) |

Values such as `0.0` and `inf` are read from memory so the compiler cannot constant-fold the
arithmetic. FP16 and BF16 results are `half2`/`bfloat162` on purpose: scalar 16-bit stores
(`STG.E.U16`) are not instrumented by nixnan (see `../README.md`).

## Collected exceptions (RTX 3090, sm_86, CUDA 12.4)

- `hello_nixnan.report.txt` — the end-of-run report (what the plain command above prints)
- `exceptions.md` — every exception grouped by kernel and SASS instruction, with the operand number
  (`0` = destination, `1..n` = sources) and decoded type
- `hello_nixnan.detail.log` — raw stderr with `PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1`
- `hello_nixnan.expect` — the baseline `../run.py` compares against

Regenerate the three collected files with `python3 collect_exceptions.py [path/to/nixnan.so]`.

Summary of the report:

| section | NaN | Inf | -Inf | Subnormal | Div by 0 |
|---|---|---|---|---|---|
| FP16 | 1 | 4 | 0 | 0 | 0 |
| BF16 | 1 | 3 | 0 | 0 | 0 |
| FP32 | 3 | 3 | 0 | 1 | 1 |
| FP64 | 17 | 2 | 0 | 0 | 1 |
| FP16 mem / BF16 mem / FP32 mem / FP64 mem (NaN) | 1 / 1 / 3 / 1 | | | | |

The FP64 NaN count of 17 is mostly not the program's own `inf * 0`: the FP64 division `1.0 / 0.0`
expands into a sequence of `DFMA`s (visible in `exceptions.md`) and the infinite reciprocal propagates
NaN through them. That makes a good talking point: one source-level exception becomes many
instruction-level ones.
