#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
/**
 * @brief CPU implementation for Linear
 * @param c Output pointer (Y)
 * @param a Input pointer (X)
 * @param w Weight pointer (W)
 * @param b Bias pointer (b), can be nullptr
 * @param type Data type
 * @param M Batch size (rows of X)
 * @param N Output features (rows of W)
 * @param K Input features (cols of X and cols of W)
 */
void linear(std::byte *c, const std::byte *a, const std::byte *w, const std::byte *b, 
            llaisysDataType_t type, size_t M, size_t N, size_t K);
}