#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// Template implementation
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    // Naive Matrix Multiplication: O(M * N * K)
    // Y[m, n] = dot(X[m, :], W[n, :]) + b[n]
    
    // Parallelize outer loops if OMP is enabled (optional/implied context)
    // #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            
            // Dot product
            // in is [M, K], accessed as in[m * K + k]
            // weight is [N, K], accessed as weight[n * K + k]
            for (size_t k = 0; k < K; ++k) {
                float x_val = 0.f;
                float w_val = 0.f;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                     x_val = llaisys::utils::cast<float>(in[m * K + k]);
                     w_val = llaisys::utils::cast<float>(weight[n * K + k]);
                } else {
                     x_val = static_cast<float>(in[m * K + k]);
                     w_val = static_cast<float>(weight[n * K + k]);
                }
                
                sum += x_val * w_val;
            }

            // Add bias if provided
            if (bias) {
                float b_val = 0.f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    b_val = llaisys::utils::cast<float>(bias[n]);
                } else {
                    b_val = static_cast<float>(bias[n]);
                }
                sum += b_val;
            }

            // Write back to output
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[m * N + n] = llaisys::utils::cast<T>(sum);
            } else {
                out[m * N + n] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *c, const std::byte *a, const std::byte *w, const std::byte *b, 
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(c), 
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(w), 
            reinterpret_cast<const float *>(b), 
            M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(c), 
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(w),
            reinterpret_cast<const llaisys::bf16_t *>(b), 
            M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(c), 
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(w),
            reinterpret_cast<const llaisys::fp16_t *>(b), 
            M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu