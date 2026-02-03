#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <type_traits>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    // Parallelize loop
    // #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float val_gate = 0.0f;
        float val_up = 0.0f;

        // Load and convert to float
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val_gate = llaisys::utils::cast<float>(gate[i]);
            val_up = llaisys::utils::cast<float>(up[i]);
        } else {
            val_gate = static_cast<float>(gate[i]);
            val_up = static_cast<float>(up[i]);
        }

        // Calculate SiLU(gate) = gate / (1 + exp(-gate))
        float silu = val_gate / (1.0f + std::exp(-val_gate));

        // Calculate result = up * SiLU(gate)
        float res = val_up * silu;

        // Store result
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(res);
        } else {
            out[i] = static_cast<T>(res);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(
            reinterpret_cast<float *>(out), 
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), 
            numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(
            reinterpret_cast<llaisys::bf16_t *>(out), 
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up), 
            numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(
            reinterpret_cast<llaisys::fp16_t *>(out), 
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up), 
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu