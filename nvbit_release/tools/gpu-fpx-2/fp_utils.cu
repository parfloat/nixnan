#ifndef FP_UTILS_H
#define FP_UTILS_H

#include <cstdint>
#include "common.h"


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
#endif // FP_UTILS_H