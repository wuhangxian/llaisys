#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"
#include <type_traits>

// Recursive function to handle N-dimensional iteration
template <typename T>
void rearrange_recursive(T *out_ptr, const T *in_ptr,
                         const std::vector<size_t>& shape,      // shape 是 size_t
                         const std::vector<int64_t>& out_strides, // strides 是 int64_t
                         const std::vector<int64_t>& in_strides, // strides 是 int64_t
                         size_t ndim,
                         size_t current_dim,
                         int64_t out_offset,
                         int64_t in_offset) {
    
    // Base case: If we are at the last dimension, iterate and copy
    if (current_dim == ndim - 1) {
        size_t len = shape[current_dim];
        int64_t s_out = out_strides[current_dim];
        int64_t s_in = in_strides[current_dim];

        for (size_t i = 0; i < len; ++i) {
            // Copy value
            out_ptr[out_offset + i * s_out] = in_ptr[in_offset + i * s_in];
        }
        return;
    }

    // Recursive step: Iterate over current dimension and go deeper
    size_t len = shape[current_dim];
    int64_t s_out = out_strides[current_dim];
    int64_t s_in = in_strides[current_dim];

    for (size_t i = 0; i < len; ++i) {
        rearrange_recursive(out_ptr, in_ptr, 
                            shape, out_strides, in_strides, 
                            ndim, current_dim + 1,
                            out_offset + i * s_out,
                            in_offset + i * s_in);
    }
}

template <typename T>
void rearrange_(T *out, const T *in,
                const std::vector<size_t>& shape,
                const std::vector<int64_t>& out_strides,
                const std::vector<int64_t>& in_strides,
                size_t ndim) {
    
    // Handle 0-dim tensor (scalar) edge case
    if (ndim == 0) {
        out[0] = in[0];
        return;
    }

    rearrange_recursive(out, in, shape, out_strides, in_strides, ndim, 0, 0, 0);
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, 
               llaisysDataType_t type, 
               const std::vector<size_t>& shape,
               const std::vector<int64_t>& out_strides,
               const std::vector<int64_t>& in_strides,
               size_t ndim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(
            reinterpret_cast<float *>(out), 
            reinterpret_cast<const float *>(in),
            shape, out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(
            reinterpret_cast<llaisys::bf16_t *>(out), 
            reinterpret_cast<const llaisys::bf16_t *>(in),
            shape, out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_(
            reinterpret_cast<llaisys::fp16_t *>(out), 
            reinterpret_cast<const llaisys::fp16_t *>(in),
            shape, out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_I64:
         return rearrange_(
            reinterpret_cast<int64_t *>(out), 
            reinterpret_cast<const int64_t *>(in),
            shape, out_strides, in_strides, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu