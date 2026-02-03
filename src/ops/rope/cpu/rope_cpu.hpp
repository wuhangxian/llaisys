#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
/**
 * @brief CPU implementation for RoPE
 * @param out Output pointer
 * @param in Input pointer
 * @param pos_ids Pointer to position IDs (int64)
 * @param type Data type of tensors
 * @param L Sequence length (dim 0)
 * @param H Number of heads (dim 1)
 * @param D Head dimension (dim 2)
 * @param theta Base frequency
 */
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, 
          llaisysDataType_t type, size_t L, size_t H, size_t D, float theta);
}
