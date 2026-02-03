#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>


template <typename T_DATA, typename T_IDX>
void embedding_(T_DATA *out, const T_IDX *index, const T_DATA *weight, 
                size_t num_indices, size_t embedding_dim, size_t vocab_size) {
    
    // 遍历每一个索引
    for (size_t i = 0; i < num_indices; i++) {
        T_IDX idx = index[i];
        
        // 简单的越界检查 (可选，但在Debug模式下很有用)
        // 生产环境为了性能通常会省略，或者使用 ASSERT
        if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) {
            // 这里简单处理，越界可能导致 segfault，实际工程中应抛出异常
            continue; 
        }

        const T_DATA* src_row = weight + idx * embedding_dim;
        T_DATA* dst_row = out + i * embedding_dim;

        // Embedding 就是整行拷贝
        // 使用 memcpy 效率最高，因为它利用了 SIMD
        std::memcpy(dst_row, src_row, embedding_dim * sizeof(T_DATA));
    }
}

namespace llaisys::ops::cpu {

void embedding(std::byte *c, const std::byte *idx, const std::byte *w, 
               llaisysDataType_t type, llaisysDataType_t index_type, 
               size_t num_indices, size_t embedding_dim, size_t vocab_size) {

    // 双重 Switch 分发：先分发数据类型，再分发索引类型
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        switch (index_type) {
        case LLAISYS_DTYPE_I32:
            return embedding_(reinterpret_cast<float *>(c), reinterpret_cast<const int32_t *>(idx), 
                              reinterpret_cast<const float *>(w), num_indices, embedding_dim, vocab_size);
        case LLAISYS_DTYPE_I64:
            return embedding_(reinterpret_cast<float *>(c), reinterpret_cast<const int64_t *>(idx), 
                              reinterpret_cast<const float *>(w), num_indices, embedding_dim, vocab_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        switch (index_type) {
        case LLAISYS_DTYPE_I32:
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const int32_t *>(idx), 
                              reinterpret_cast<const llaisys::bf16_t *>(w), num_indices, embedding_dim, vocab_size);
        case LLAISYS_DTYPE_I64:
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const int64_t *>(idx), 
                              reinterpret_cast<const llaisys::bf16_t *>(w), num_indices, embedding_dim, vocab_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        switch (index_type) {
        case LLAISYS_DTYPE_I32:
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const int32_t *>(idx), 
                              reinterpret_cast<const llaisys::fp16_t *>(w), num_indices, embedding_dim, vocab_size);
        case LLAISYS_DTYPE_I64:
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const int64_t *>(idx), 
                              reinterpret_cast<const llaisys::fp16_t *>(w), num_indices, embedding_dim, vocab_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu