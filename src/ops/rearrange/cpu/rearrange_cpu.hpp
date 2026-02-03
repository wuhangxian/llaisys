#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>
#include <cstdint>

namespace llaisys::ops::cpu {

/**
 * @brief CPU implementation for Rearrange
 * @param out Output data pointer
 * @param in Input data pointer
 * @param type Data type
 * @param shape Shape vector (std::vector<size_t>)
 * @param out_strides Output strides vector (std::vector<int64_t>)
 * @param in_strides Input strides vector (std::vector<int64_t>)
 * @param ndim Number of dimensions
 */
void rearrange(std::byte *out, const std::byte *in, 
               llaisysDataType_t type, 
               const std::vector<size_t>& shape,      // 注意：这里用 size_t
               const std::vector<int64_t>& out_strides, // 注意：这里用 int64_t
               const std::vector<int64_t>& in_strides, // 注意：这里用 int64_t
               size_t ndim);

} // namespace llaisys::ops::cpu