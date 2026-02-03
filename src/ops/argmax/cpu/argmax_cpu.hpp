#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte* max_idx_ptr, std::byte* max_val_ptr, const std::byte* vals_ptr, llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::cpu