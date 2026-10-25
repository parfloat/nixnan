# hello_nixnan: exceptions nixnan found

Collected by `collect_exceptions.py` (PRINT_ILL_INSTR=1). `operand 0` is the destination, `1..n` are the sources; `type` is the operand type nixnan decoded. An `STG` row is the memory path: a NaN being stored to global memory.


## kernel `hello`

| instruction | type | exceptions (operand) |
|---|---|---|
| `MUFU.RCP64H R11, R7` | f64 | div0 (op 1), infinity (op 0) |
| `DFMA R12, -R6, R10, 1` | f64 | NaN (op 2), NaN (op 0) |
| `DFMA R12, R12, R12, R12` | f64 | NaN (op 1), NaN (op 2), NaN (op 3), NaN (op 0) |
| `DFMA R12, R10, R12, R10` | f64 | NaN (op 1), NaN (op 2), NaN (op 3), NaN (op 0) |
| `DFMA R14, -R6, R12, 1` | f64 | NaN (op 2), NaN (op 0) |
| `DFMA R12, R12, R14, R12` | f64 | NaN (op 1), NaN (op 2), NaN (op 3), NaN (op 0) |
| `DMUL R14, R6, R4` | f64 | infinity (op 2), NaN (op 0) |
| `STG.E.64 [R10.64+0x8], R14` | f64 | NaN (op 1) |
| `@!P0 MUFU.RCP R6, R0` | f32 | div0 (op 1), infinity (op 0) |
| `HMUL2 R0, R0.H0_H0, 65504, 65504` | f16 | infinity (op 0) |
| `FMUL R11, R8, R5` | f32 | infinity (op 2), NaN (op 0) |
| `STG.E [R6.64+0x4], R11` | f32 | NaN (op 1) |
| `HMUL2 R9, R0, R4.H0_H0` | f16 | infinity (op 1), NaN (op 0) |
| `FMUL R13, R2, 1.175494350822287508e-38` | f32 | subnormal (op 0) |
| `STG.E [R2.64+0x4], R9` | f16 | NaN (op 1) |
| `HFMA2.BF16_V2 R7, R5.H0_H0, 3.00405527047390987276e+38, 3.00405527047390987276e+38, -RZ.H0_H0` | bf16 | infinity (op 1), infinity (op 0) |
| `HFMA2.BF16_V2 R15, R7, R8.H0_H0, -RZ.H0_H0` | bf16 | infinity (op 1), NaN (op 0) |
| `STG.E [R4.64+0x4], R15` | bf16 | NaN (op 1) |

## kernel `tc`

| instruction | type | exceptions (operand) |
|---|---|---|
| `HMMA.16816.F32 R16, R8.reuse, R12, RZ` | f16 | infinity (op 1) |
| `HMMA.16816.F32 R16, R8.reuse, R12, RZ` | f32 | NaN,infinity (op 0) |
| `HMMA.16816.F32 R12, R8, R14, RZ` | f16 | infinity (op 1) |
| `HMMA.16816.F32 R12, R8, R14, RZ` | f32 | NaN (op 0) |
| `STG.E.64 [R8.64], R16` | f32 | NaN (op 1) |
| `STG.E.64 [R8.64+0x20], R12` | f32 | NaN (op 1) |
