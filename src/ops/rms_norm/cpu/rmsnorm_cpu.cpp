#include "rmsnorm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    // Iterate over each row (sample)
    for (size_t i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;
        
        // Offset for the current row
        const T* row_in = in + i * cols;
        T* row_out = out + i * cols;

        // 1. Calculate Sum of Squares
        for (size_t j = 0; j < cols; ++j) {
            float val = 0.f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(row_in[j]);
            } else {
                val = static_cast<float>(row_in[j]);
            }
            sum_sq += val * val;
        }

        // 2. Calculate RMS and Inverse RMS
        // rms = sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        float inv_rms = 1.0f / rms;

        // 3. Normalize and Scale
        // y = (x * inv_rms) * w
        for (size_t j = 0; j < cols; ++j) {
            float val = 0.f;
            float w_val = 0.f;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(row_in[j]);
                w_val = llaisys::utils::cast<float>(weight[j]);
            } else {
                val = static_cast<float>(row_in[j]);
                w_val = static_cast<float>(weight[j]);
            }

            float result = val * inv_rms * w_val;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                row_out[j] = llaisys::utils::cast<T>(result);
            } else {
                row_out[j] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *c, const std::byte *a, const std::byte *w, llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(c), 
            reinterpret_cast<const float *>(a), 
            reinterpret_cast<const float *>(w), 
            rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(c), 
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(w), 
            rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(c), 
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(w), 
            rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu