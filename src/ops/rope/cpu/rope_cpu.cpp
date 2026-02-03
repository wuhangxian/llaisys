#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t L, size_t H, size_t D, float theta) {
    size_t half_D = D / 2;

    // Iterate over Sequence Length
    for (size_t i = 0; i < L; ++i) {
        for (size_t h = 0; h < H; ++h) {
            
            int64_t pos = pos_ids[i];
            
            // Base offset for this token and this head
            size_t base_offset = (i * H * D) + (h * D);

            // Iterate over half the dimension
            for (size_t j = 0; j < half_D; ++j) {
                // ====================================================
                // 【关键修改】：使用 double 进行中间频率和角度计算
                // ====================================================
                
                // 计算 exponent: 2j / d
                double freq_exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(D);
                
                // 计算 inv_freq: theta ^ (-2j/d)
                // 使用 double 避免 pow 的精度损失
                double inv_freq = 1.0 / std::pow(static_cast<double>(theta), freq_exponent);
                
                // 计算角度: pos * inv_freq
                // pos 是 int64，直接乘 float 会损失精度，必须乘 double
                double angle = static_cast<double>(pos) * inv_freq;

                // 使用 double精度的 cos/sin
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);

                // ====================================================

                // Indices for the pair [a, b]
                size_t idx_a = base_offset + j;
                size_t idx_b = base_offset + j + half_D;

                // Load values (Cast input to float for calculation)
                float val_a = 0.f;
                float val_b = 0.f;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    val_a = llaisys::utils::cast<float>(in[idx_a]);
                    val_b = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    val_a = static_cast<float>(in[idx_a]);
                    val_b = static_cast<float>(in[idx_b]);
                }

                // Apply rotation
                // 计算时使用 double 的 cos/sin 结果，但在乘法前可以转回 float，
                // 或者全程 double 计算后再转回 T。
                // 为了最高精度对齐，建议运算也用 double，最后再转 T
                double res_a = static_cast<double>(val_a) * cos_val - static_cast<double>(val_b) * sin_val;
                double res_b = static_cast<double>(val_b) * cos_val + static_cast<double>(val_a) * sin_val;

                // Store values
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(static_cast<float>(res_a));
                    out[idx_b] = llaisys::utils::cast<T>(static_cast<float>(res_b));
                } else {
                    out[idx_a] = static_cast<T>(res_a);
                    out[idx_b] = static_cast<T>(res_b);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, 
          llaisysDataType_t type, size_t L, size_t H, size_t D, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out), 
            reinterpret_cast<const float *>(in), 
            pos_ids, L, H, D, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out), 
            reinterpret_cast<const llaisys::bf16_t *>(in), 
            pos_ids, L, H, D, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out), 
            reinterpret_cast<const llaisys::fp16_t *>(in), 
            pos_ids, L, H, D, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu