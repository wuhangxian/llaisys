#include "selfattention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len,
                     size_t nhead, size_t nkvhead,
                     size_t d, size_t dv,
                     float scale) {
    
    // Group size for GQA
    size_t group_size = nhead / nkvhead;

    // Buffer for attention scores (logits)
    std::vector<float> logits(total_len);

    // Iterate over sequence length (Query tokens)
    for (size_t i = 0; i < seqlen; ++i) {
        
        size_t query_global_pos = (total_len - seqlen) + i;

        // Iterate over each Query Head
        for (size_t h = 0; h < nhead; ++h) {
            
            size_t kv_h = h / group_size;

            // Pointers
            const T* q_vec = q + (i * nhead * d) + (h * d);
            
            // --- Step 1: Calculate Attention Scores (Q * K^T) ---
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t t = 0; t < total_len; ++t) {
                if (t > query_global_pos) {
                    logits[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const T* k_vec = k + (t * nkvhead * d) + (kv_h * d);
                
                float score = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    float q_val = 0.f;
                    float k_val = 0.f;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_vec[j]);
                        k_val = llaisys::utils::cast<float>(k_vec[j]);
                    } else {
                        q_val = static_cast<float>(q_vec[j]);
                        k_val = static_cast<float>(k_vec[j]);
                    }
                    score += q_val * k_val;
                }
                
                score *= scale;
                logits[t] = score;
                
                if (score > max_score) {
                    max_score = score;
                }
            }

            // --- Step 2: Softmax ---
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (t > query_global_pos) {
                    logits[t] = 0.0f; 
                } else {
                    float exp_val = std::exp(logits[t] - max_score);
                    logits[t] = exp_val;
                    sum_exp += exp_val;
                }
            }
            
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            // --- Step 3: Weighted Sum (Prob * V) ---
            // 【关键修改】：使用 float 类型的临时缓冲区 acc_out 进行累加
            // 避免在循环中反复转换回 fp16 导致精度丢失
            std::vector<float> acc_out(dv, 0.0f);

            for (size_t t = 0; t < total_len; ++t) {
                if (logits[t] == 0.0f) continue;

                float prob = logits[t] * inv_sum;
                
                const T* v_vec = v + (t * nkvhead * dv) + (kv_h * dv);

                for (size_t j = 0; j < dv; ++j) {
                    float v_val = 0.f;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v_vec[j]);
                    } else {
                        v_val = static_cast<float>(v_vec[j]);
                    }
                    
                    // 始终在 float 精度下累加
                    acc_out[j] += prob * v_val;
                }
            }

            // 【关键修改】：计算完成后，一次性写回 Output 内存
            T* out_vec = out + (i * nhead * dv) + (h * dv);
            for (size_t j = 0; j < dv; ++j) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_vec[j] = llaisys::utils::cast<T>(acc_out[j]);
                } else {
                    out_vec[j] = static_cast<T>(acc_out[j]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, 
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv, 
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu