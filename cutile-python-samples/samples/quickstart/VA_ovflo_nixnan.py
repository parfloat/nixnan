# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
FP16 vector-add overflow sample for NixNan experiments.

The inputs are finite FP16 values below the largest normal FP16 value, but the
sum is outside the FP16 finite range. The expected output is +inf in every lane.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


FP16_MAX_NORMAL = np.float16(65504.0)
OVERFLOW_INPUT = np.float16(60000.0)


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid.
    pid = ct.bid(0)

    # Load FP16 input tiles.
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # FP16 addition overflows because 60000 + 60000 exceeds 65504.
    result = a_tile + b_tile

    # Store the overflowing FP16 result.
    ct.store(c, index=(pid,), tile=result)


def test():
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.full(vector_size, OVERFLOW_INPUT, dtype=cp.float16)
    b = cp.full(vector_size, OVERFLOW_INPUT, dtype=cp.float16)
    c = cp.zeros(vector_size, dtype=cp.float16)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        vector_add,
        (a, b, c, tile_size),
    )

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    assert a_np.dtype == np.float16
    assert b_np.dtype == np.float16
    assert c_np.dtype == np.float16
    assert np.all(np.isfinite(a_np))
    assert np.all(np.isfinite(b_np))
    assert np.all(a_np < FP16_MAX_NORMAL)
    assert np.all(b_np < FP16_MAX_NORMAL)
    assert np.all(np.isposinf(c_np))

    print("fp16_vector_add_overflow_nixnan passed")


if __name__ == "__main__":
    test()
