#ifndef FP_UTILS_CUH
#define FP_UTILS_CUH

#include <cstdint>
#include "common.cuh"


// Half utils
__device__ __host__ inline
uint16_t half_exp(uint16_t h) {
    return (h & 0x7C00) >> 10;
}

__device__ __host__ inline
uint16_t half_mant(uint16_t h) {
    return h & 0x03FF;
}

__device__ __host__ inline
uint16_t half_sign(uint16_t h) {
    return h >> 15;
}

__device__ __host__ inline
uint32_t half_is_nan(uint16_t h) {
    return (half_exp(h) == 0x1F && half_mant(h) != 0) ? E_NAN : 0;
}

__device__ __host__ inline
uint32_t half_is_inf(uint16_t h) {
    return (half_exp(h) == 0x1F && half_mant(h) == 0) ? E_INF : 0;
}

__device__ __host__ inline
uint32_t half_is_zero(uint16_t h) {
    return (half_exp(h) == 0 && half_mant(h) == 0) ? E_DIV0 : 0;
}

__device__ __host__ inline
uint32_t half_is_subnorm(uint16_t h) {
    return (half_exp(h) == 0 && half_mant(h) != 0) ? E_SUB : 0;
}

__device__ __host__ inline
uint32_t half_classify(uint16_t h, bool is_zero = false) {
    uint32_t e = half_is_nan(h);
    e |= half_is_inf(h);
    e |= half_is_subnorm(h);
    if (is_zero) {
        e |= half_is_zero(h);
    }
    return e;
}

// Half2 utils
__device__ __host__ inline
uint32_t half2_is_nan(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? half_is_nan(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? half_is_nan(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t half2_is_inf(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? half_is_inf(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? half_is_inf(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t half2_is_zero(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? half_is_zero(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? half_is_zero(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t half2_is_subnorm(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? half_is_subnorm(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? half_is_subnorm(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t half2_classify(uint32_t h, bool h0, bool h1, bool is_zero = false) {
    uint32_t e0 = half2_is_nan(h, h0, h1);
    e0 |= half2_is_inf(h, h0, h1);
    e0 |= half2_is_subnorm(h, h0, h1);
    if (is_zero) {
        e0 |= half2_is_zero(h, h0, h1);
    }
    return e0;
}

// BF16 utils
__device__ __host__ inline
uint16_t bf16_exp(uint16_t h) {
    return (h & 0x7F80) >> 7;
}

__device__ __host__ inline
uint16_t bf16_mant(uint16_t h) {
    return h & 0x007F;
}

__device__ __host__ inline
uint16_t bf16_sign(uint16_t h) {
    return h >> 15;
}

__device__ __host__ inline
uint32_t bf16_is_nan(uint16_t h) {
    return (bf16_exp(h) == 0xFF && bf16_mant(h) != 0) ? E_NAN : 0;
}

__device__ __host__ inline
uint32_t bf16_is_inf(uint16_t h) {
    return (bf16_exp(h) == 0xFF && bf16_mant(h) == 0) ? E_INF : 0;
}

__device__ __host__ inline
uint32_t bf16_is_zero(uint16_t h) {
    return (bf16_exp(h) == 0 && bf16_mant(h) == 0) ? E_DIV0 : 0;
}

__device__ __host__ inline
uint32_t bf16_is_subnorm(uint16_t h) {
    return (bf16_exp(h) == 0 && bf16_mant(h) != 0) ? E_SUB : 0;
}

__device__ __host__ inline
uint32_t bf16_classify(uint16_t h, bool is_zero = false) {
    uint32_t e = bf16_is_nan(h);
    e |= bf16_is_inf(h);
    e |= bf16_is_subnorm(h);
    if (is_zero) {
        e |= bf16_is_zero(h);
    }
    return e;
}

// bf162 utils
__device__ __host__ inline
uint32_t bf162_is_nan(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? bf16_is_nan(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? bf16_is_nan(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t bf162_is_inf(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? bf16_is_inf(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? bf16_is_inf(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t bf162_is_zero(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? bf16_is_zero(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? bf16_is_zero(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t bf162_is_subnorm(uint32_t h, bool h0, bool h1) {
    uint8_t e0 = h0 ? bf16_is_subnorm(h & 0xFFFF) : 0;
    uint8_t e1 = h1 ? bf16_is_subnorm(h >> 16) : 0;
    return e0 | e1;
}

__device__ __host__ inline
uint32_t bf162_classify(uint32_t h, bool h0, bool h1, bool is_zero = false) {
    uint32_t e0 = bf162_is_nan(h, h0, h1);
    e0 |= bf162_is_inf(h, h0, h1);
    e0 |= bf162_is_subnorm(h, h0, h1);
    if (is_zero) {
        e0 |= bf162_is_zero(h, h0, h1);
    }
    return e0;
}

// Float utils
__device__ __host__ inline
uint32_t float_exp(uint32_t f) {
    return (f & 0x7F80'0000) >> 23;
}

__device__ __host__ inline
uint32_t float_mant(uint32_t f) {
    return f & 0x007F'FFFF;
}

__device__ __host__ inline
uint32_t float_is_nan(uint32_t f) {
    return (float_exp(f) == 0xFF && float_mant(f) != 0) ? E_NAN : 0;
}

__device__ __host__ inline
uint32_t float_is_inf(uint32_t f) {
    return (float_exp(f) == 0xFF && float_mant(f) == 0) ? E_INF : 0;
}

__device__ __host__ inline
uint32_t float_is_zero(uint32_t f) {
    return (float_exp(f) == 0 && float_mant(f) == 0) ? E_DIV0 : 0;
}

__device__ __host__ inline
uint32_t float_is_subnorm(uint32_t f) {
    return (float_exp(f) == 0 && float_mant(f) != 0) ? E_SUB : 0;
}

__device__ __host__ inline
uint32_t float_classify(uint32_t f, bool is_zero = false) {
    uint32_t e = float_is_nan(f);
    e |= float_is_inf(f);
    e |= float_is_subnorm(f);
    if (is_zero) {
        e |= float_is_zero(f);
    }
    return e;
}

// Double utils
__device__ __host__ inline
uint32_t double_exp(uint32_t low, uint32_t high) {
    return (((uint64_t)high << 32 | low) & 0x7FF0'0000'0000'0000) >> 52;
}

__device__ __host__ inline
uint64_t double_mant(uint32_t low, uint32_t high) {
    return ((uint64_t)high << 32 | low) & 0x000F'FFFF'FFFF'FFFF;
}

__device__ __host__ inline
uint32_t double_is_nan(uint32_t low, uint32_t high) {
    return (double_exp(low, high) == 0x7FF && double_mant(low, high) != 0) ? E_NAN : 0;
}

__device__ __host__ inline
uint32_t double_is_inf(uint32_t low, uint32_t high) {
    return (double_exp(low, high) == 0x7FF && double_mant(low, high) == 0) ? E_INF : 0;
}

__device__ __host__ inline
uint32_t double_is_zero(uint32_t low, uint32_t high) {
    return (double_exp(low, high) == 0 && double_mant(low, high) == 0) ? E_DIV0 : 0;
}

__device__ __host__ inline
uint32_t double_is_subnorm(uint32_t low, uint32_t high) {
    return (double_exp(low, high) == 0 && double_mant(low, high) != 0) ? E_SUB : 0;
}

__device__ __host__ inline
uint32_t double_classify(uint32_t low, uint32_t high, bool is_zero = false) {
    uint32_t e = double_is_nan(low, high);
    e |= double_is_inf(low, high);
    e |= double_is_subnorm(low, high);
    if (is_zero) {
        e |= double_is_zero(low, high);
    }
    return e;
}
#endif // FP_UTILS_CUH