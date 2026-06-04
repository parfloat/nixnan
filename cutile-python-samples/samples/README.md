<!--
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

**cuTile Code Samples**

This repository contains various examples demonstrating the use of cuTile for implementing high-performance GPU kernels in Python. cuTile simplifies writing CUDA kernels by providing a Pythonic interface for GPU programming concepts like tiling, shared memory, and warp-level operations.

Each sample showcases a fundamental operation, implemented directly using cuTile kernel.

**Samples Included**

**Batch Matrix Multiplication** (BatchMatMul.py)

Purpose: Implements batched matrix multiplication (C = A * B) for 3D tensors.

Key Concepts: 3D grid launches, ct.mma for efficient matrix multiply-accumulate.

Dependencies: torch, math, numpy

**Fast Fourier Transform (FFT)** (FFT.py)

Purpose: Implements a Batched 1D FFT using a multi-dimensional factorization approach.

Key Concepts: Tensor factorization, complex arithmetic, pre-computed rotation (W) and twiddle (T) factors.

Dependencies: torch, math

**Matrix Multiplication** (MatMul.py)

Purpose: Implements standard (non-batched) matrix multiplication (C = A * B) for 2D matrices.

Key Concepts: Tiled processing, efficient inner loop computation.

Dependencies: torch, math

**Matrix Transposition** (Transpose.py)

Purpose: Demonstrates transposing a 2D matrix.

Key Concepts: Tiled processing, index swapping for transposition.

Dependencies: torch, math

**Attention Fused Multi-Head Attention** (AttentionFMHA.py)

Purpose: Demonstrates a fused multi-head attention operation, common in transformer models.

Key Concepts: Casual and Non-Casual Attention

Dependencies: torch, math, numpy 
