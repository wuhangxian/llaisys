#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
/**
 * @brief CPU implementation for RMS Norm
 * @param c Output pointer
 * @param a Input pointer
 * @param w Weight pointer
 * @param type Data type
 * @param rows Number of rows (Batch size, M)
 * @param cols Number of columns (Hidden dim, d)
 * @param eps Epsilon
 */
void rms_norm(std::byte *c, const std::byte *a, const std::byte *w, llaisysDataType_t type, size_t rows, size_t cols, float eps);
}