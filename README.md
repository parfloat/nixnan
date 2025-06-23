# About
A tool designed to provide a new framework for floating-point exceptional-value detection.
The tool uses [NVbit](https://github.com/NVlabs/NVBit) in order to instrument Nvidia CUDA programs.
The goals are to add more support for instructions (and instruction types) from [GPU-FPX]() in a more general framework that will allow new instrumentation passes to be added.

# Requirements
Linux on x86 with Cuda version 12 and compute capability greater than or equal to 8.6.
ARM should work, but will replacing the x86 version of NVbit downloaded by `make`.

# Setup
Run `make` in the root directory.
This should produce a file in the main director called `nixnan.so`.
This is the instrumentation library.
(The older detector and analyzer are located in nvbit_release/tools/GPU-FPX.)

# Usage
In the `examples` directory, run the following command:
```nvcc -arch=compute_86 -lineinfo basic.cu -o basic```
Then run:
```LD_PRELOAD=../nixnan.so basic```
You should then get something like the following:
```
----- Test overflow: A[0,0]=max_normal, B[0,0]=max_normal, C[0,0]=0 -----
A[0,0] = 65504.000000 (0x7bff), B[0,0] = 65504.000000 (0x7bff), C[0,0] = 0.000000 (0x0000)
Computing D = A * B + C with Tensor Cores...
#nixnan: Initializing GPU context...
#nixnan: Could not open kernel_whitelist.txt!
#nixnan: Could not open kernel_blacklist.txt!
#nixnan: instrumenting all kernels
#nixnan: running kernel [WMMAF16TensorCore] ...
#nixnan: error [infinity] detected in instruction HMMA.16816.F16 R20, R4.reuse, R16, RZ ; in function WMMAF16TensorCore at line 0 of type f16
#nixnan: error [infinity] detected in instruction HMMA.16816.F16 R8, R4.reuse, R16, R8 ; in function WMMAF16TensorCore at line 0 of type f16
D[0,0]=inf (0x7c00)
--------------------------------------------------------------------------------
----- Test NaN: A[0,0]=inf, B[0,0]=1, C[0,0]=-inf -----
A[0,0] = inf (0x7c00), B[0,0] = 1.000000 (0x3c00), C[0,0] = -inf (0xfc00)
Computing D = A * B + C with Tensor Cores...
#nixnan: error [NaN,infinity] detected in instruction HMMA.16816.F16 R20, R4.reuse, R16, RZ ; in function WMMAF16TensorCore at line 0 of type f16
#nixnan: error [NaN] detected in instruction HMMA.16816.F16 R22, R4, R18, RZ ; in function WMMAF16TensorCore at line 0 of type f16
D[0,0]=nan (0x7fff)
--------------------------------------------------------------------------------
----- Test underflow mul: A[0,0]=min_normal, B[0,0]=0.5, C[0,0]=0.0 -----
A[0,0] = 0.000061 (0x0400), B[0,0] = 0.500000 (0x3800), C[0,0] = 0.000000 (0x0000)
Computing D = A * B + C with Tensor Cores...
#nixnan: error [subnormal] detected in instruction HMMA.16816.F16 R20, R4.reuse, R16, RZ ; in function WMMAF16TensorCore at line 0 of type f16
#nixnan: error [subnormal] detected in instruction HMMA.16816.F16 R8, R4.reuse, R16, R8 ; in function WMMAF16TensorCore at line 0 of type f16
D[0,0]=0.000031 (0x0200)
--------------------------------------------------------------------------------
----- Test underflow mul: A[0,0]=neg_min_normal, B[0,0]=0.5, C[0,0]=0.0 -----
A[0,0] = -0.000061 (0x8400), B[0,0] = 0.500000 (0x3800), C[0,0] = 0.000000 (0x0000)
Computing D = A * B + C with Tensor Cores...
D[0,0]=-0.000031 (0x8200)
--------------------------------------------------------------------------------
----- Test underflow mul: A[0,0]=min_subnormal, B[0,0]=0.5, C[0,0]=0.0 -----
A[0,0] = 0.000000 (0x0001), B[0,0] = 0.500000 (0x3800), C[0,0] = 0.000000 (0x0000)
Computing D = A * B + C with Tensor Cores...
D[0,0]=0.000000 (0x0000)
--------------------------------------------------------------------------------
----- Test underflow mul: A[0,0]=neg_min_subnormal, B[0,0]=0.5, C[0,0]=0.0 -----
A[0,0] = -0.000000 (0x8001), B[0,0] = 0.500000 (0x3800), C[0,0] = 0.000000 (0x0000)
Computing D = A * B + C with Tensor Cores...
D[0,0]=0.000000 (0x0000)
--------------------------------------------------------------------------------
#nixnan: Finalizing GPU context...

#nixnan: ------------ nixnan Report -----------

#nixnan: --- FP16 Operations ---
#nixnan: NaN: 4
#nixnan: Infinity: 3
#nixnan: Subnormal: 2
#nixnan: Division by 0: 0

#nixnan: --- FP32 Operations ---
#nixnan: NaN: 0
#nixnan: Infinity: 0
#nixnan: Subnormal: 0
#nixnan: Division by 0: 0

#nixnan: --- FP64 Operations ---
#nixnan: NaN: 0
#nixnan: Infinity: 0
#nixnan: Subnormal: 0
#nixnan: Division by 0: 0
```

The tool notifies the user of detected errors in the program.
For example in:
`#nixnan: error [infinity] detected in instruction HMMA.16816.F16 R20, R4.reuse, R16, RZ ; in function WMMAF16TensorCore at line 0 of type f16`
An infinity was detected arising in a 16-bit matrix-multiply-and-accumulate instruction.

The summary at the end indicates that there were four NaN values, three infinity values and two subnormal values generated during program execution.