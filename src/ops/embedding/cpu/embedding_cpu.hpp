#pragma once
#include "llaisys.h"

#include <cstddef>

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
// void embedding(tensor_t out, tensor_t index, tensor_t weight);
void embedding(std::byte *c, const std::byte *idx, const std::byte *w, 
               llaisysDataType_t type, llaisysDataType_t index_type, 
               size_t num_indices, size_t embedding_dim, size_t vocab_size);
}