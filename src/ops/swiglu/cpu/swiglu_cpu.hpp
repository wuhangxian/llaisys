#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
/**
 * @brief CPU implementation for SwiGLU
 * @param out Output pointer
 * @param gate Gate pointer
 * @param up Up pointer
 * @param type Data type
 * @param numel Total number of elements
 */
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::cpu