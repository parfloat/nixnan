# examples/

Two things live here. Neither is part of the regression suite (that is `../test/`).

```
examples/
├── basic.cu                    one hand-written FP16 WMMA program with injected exceptions
└── mma-examples/
    └── extractmmas.py          generator: emits a .cu with one kernel per WMMA variant
```

## basic.cu

A single 16x16x16 FP16 tile. A and B start as identity, C as 0; each of six cases overwrites
element `[0,0]` with a special value, so one tensor-core multiply-accumulate hits one exception:

| Case | A[0,0] | B[0,0] | C[0,0] | Exception |
|------|--------|--------|--------|-----------|
| 1 | 65504 (max normal) | 65504 | 0 | overflow to +inf |
| 2 | +inf | 1 | -inf | NaN (inf + -inf) |
| 3, 4 | +/- min normal | 0.5 | 0 | result is a subnormal |
| 5, 6 | +/- min subnormal | 0.5 | 0 | result rounds to zero |

It prints the inputs and `D[0,0]` only; it asserts nothing.

Build and run:

    nvcc -o basic basic.cu -arch sm_86
    ./basic                                   # native
    LD_PRELOAD=<path>/nixnan.so ./basic       # under nixnan

(The comment at the top of the file says `main.cu`/`main`; that is a leftover name.)

This file is byte-identical to `../test/mma/basic.cu`. The two are separate copies, so edit both
or neither. `../test/mma/basic.expect` is the expected nixnan report and
`../test/mma/readme_basic_cu` explains the program in more detail.

## mma-examples/extractmmas.py

A code generator. It reads `crt/mma.h` from the CUDA toolkit, collects every `mma_sync`
combination `(M, N, K, D type, A type, B type, C type)`, drops the integer and sub-byte modes
(`char`, `int`, `u4`, `b1`, `s4`), and prints a complete CUDA program to stdout containing:

- one `__global__` kernel per combination, named `mma_M_N_K_D_A_B_C`
  (load A, B, C; `mma_sync`; store D; one warp);
- host helpers: `from_double`/`to_double`, a seeded `fill_mat` (uniform [0,1)), and
  `mma_cpu_error` (double-precision reference, returns total absolute error);
- a `main()` that launches every kernel with `<<<1,32>>>` on managed memory, prints the D matrix
  and `Total error`.

Inputs are random values in [0,1), so **no exceptions are injected** and there is no `.expect`
file. Its use is to enumerate which tensor-core variants the toolkit offers and to give nixnan a
program that exercises all of them.

### Usage

    python3 extractmmas.py > mma_all.cu
    nvcc -arch sm_86 -o mma_all mma_all.cu
    ./mma_all

### Header lookup (fixed)

The script used to open a hardcoded `/usr/include/crt/mma.h`, which does not exist when CUDA is
installed under `/usr/local/cuda*`, and it crashed with `FileNotFoundError`. It now looks, in
order, at:

1. `argv[1]`  (`python3 extractmmas.py /path/to/crt/mma.h`)
2. `$MMA_H`
3. the toolkit of the `nvcc` on `PATH` (`<root>/targets/x86_64-linux/include/crt/mma.h`, then `<root>/include/crt/mma.h`)
4. `/usr/local/cuda/...` and `/usr/include/crt/mma.h`

and exits with a clear message if none exists.

### Result with CUDA 12.4 (RTX 3090, sm_86)

It generated 17 kernels (946 lines). The program compiled for `-arch sm_86` and all 17 ran
without a CUDA error:

| Shape (M_N_K) | Variants (D <- A, B, C) |
|---|---|
| 16x16x16, 32x8x16, 8x32x16 | half<-half,half,half; float<-half,half,half; half<-half,half,float; float<-half,half,float; float<-bf16,bf16,float |
| 16x16x8 | float<-tf32,tf32,float |
| 8x8x4 | double<-double,double,double |

Reported total errors against the CPU reference were about 1e-4 to 5e-5 for float-output
variants, about 0.2 for half-output variants (the CPU sum is not rounded to half), 0.35 for TF32,
and 6e-15 for the double variant. I did not examine which SASS instruction each variant compiles
to, so I cannot say whether the double one runs on tensor cores on sm_86.
