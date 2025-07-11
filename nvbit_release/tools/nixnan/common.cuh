#ifndef COMMON_CUH
#define COMMON_CUH

#include <cstdint>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <cstring>

const uint32_t E_NAN = 1,
  E_INF = 2,
  E_SUB = 4,
  E_DIV0 = 8;

const char FP16 = 0,
  BF16 = 1,
  FP32 = 2,
  FP64 = 3;

  #include <unordered_map>

const std::unordered_map<char, std::string> type_to_string = {
  {FP16, "f16"},
  {BF16, "bf16"},
  {FP32, "f32"},
  {FP64, "f64"}
};

const std::unordered_map<std::string, char> string_to_type = {
  {"f16", FP16},
  {"bf16", BF16},
  {"f32", FP32},
  {"f64", FP64}
};

std::unordered_set<std::string> read_from_file(std::string filename);

inline std::string cut_kernel_name(const std::string& kernel_name) {
  size_t pos = kernel_name.find_first_of("<(");
  if (pos != std::string::npos) {
    return kernel_name.substr(0, pos);
  }
  return kernel_name;
}

template<typename T>
uint64_t tobits64(T value) {
  static_assert(sizeof(T) == sizeof(uint64_t), "T must be 64 bits");
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

template<typename T>
uint32_t tobits32(T value) {
  static_assert(sizeof(T) == sizeof(uint32_t), "T must be 32 bits");
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

const int OPERANDS = 4;
const int OPBITS = 2;
const int EXCEBITS = 4;

#endif // COMMON_CUH