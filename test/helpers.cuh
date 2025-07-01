#include <cuda_fp16.h>
#ifndef __HELPERS_CUH__
#define __HELPERS_CUH__

void fill_array_float(float *array, size_t len, float val) {
    for (size_t i = 0; i < len; i++) {
        array[i] = val;
    }
}

void fill_array_float2(float2 *array, size_t len, float val) {
    for (size_t i = 0; i < len; i++) {
        array[i].x = val;
        array[i].y = val;
    }
}

void fill_array_double(double *array, size_t len, double val) {
    for (size_t i = 0; i < len; i++) {
        array[i] = val;
    }
}

void fill_array_half(half *array, size_t len, half val) {
    for (size_t i = 0; i < len; i++) {
        array[i] = val;
    }
}

void fill_array_half2(half2 *array, size_t len, half val) {
    for (size_t i = 0; i < len; i++) {
        array[i].x = val;
        array[i].y = val;
    }
}

__host__ __device__
inline unsigned short h2s(half h) {
    unsigned short s;
    memcpy(&s, &h, sizeof(h));
    return s;
}

__host__ __device__
inline half s2h(unsigned short s) {
    half h;
    memcpy(&h, &s, sizeof(h));
    return h;
}

__host__ __device__
inline unsigned int h22u(half2 h) {
    unsigned int u;
    memcpy(&u, &h, sizeof(h));
    return u;
}

__host__ __device__
inline half2 u2h2(unsigned int u) {
    half2 h;
    memcpy(&h, &u, sizeof(h));
    return h;
}

#define HLF_MAX 65504.0
#define HLF_MIN (1.0/16384.0)

constexpr int sat = 0;
constexpr int nosat = 1;
template <int mode>
struct saturate;

template<> struct saturate<sat> {
    static constexpr const char mode[] = ".sat";
};

template<> struct saturate<nosat> {
    static constexpr const char mode[] = "";
};

constexpr int tozero = 0;
constexpr int noflush = 1;
template <int mode>
struct flush;

template<> struct flush<tozero> {
    static constexpr const char mode[] = ".ftz";
};

template<> struct flush<noflush> {
    static constexpr const char mode[] = "";
};

constexpr int rn = 0;
constexpr int rz = 1;
constexpr int rm = 2;
constexpr int rp = 3;

template <int mode>
struct rnd;

template<> struct rnd<rn> {
    static constexpr const char mode[] = ".rn";
};

template<> struct rnd<rz> {
    static constexpr const char mode[] = ".rz";
};

template<> struct rnd<rm> {
    static constexpr const char mode[] = ".rm";
};

template<> struct rnd<rp> {
    static constexpr const char mode[] = ".rp";
};
#endif // __HELPERS_CUH__