#include "op.hpp"

#include "../../core/llaisys_core.hpp"


#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 检查 Device: out 和 weight 通常需要在同一个设备上
    // index 在很多框架中可以在 CPU，但为了简单起见，这里假设都在同一设备，或者遵循框架的 CHECK_SAME_DEVICE 逻辑
    CHECK_SAME_DEVICE(out, weight); 
    
    // 2. 检查 contiguous
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous.");

    // 3. 检查 dtype: out 和 weight 必须一致 (float/half)，index 必须是整型
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I32 || index->dtype() == LLAISYS_DTYPE_I64,
           "Embedding: index must be I32 or I64.");

    // 4. 检查 Shape
    // Weight 必须是 2D [VocabSize, EmbeddingDim]
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    size_t vocab_size = weight->shape()[0];
    size_t embedding_dim = weight->shape()[1];
    size_t num_indices = index->numel();
    
    // Out 的最后一维必须等于 EmbeddingDim
    ASSERT(out->shape().back() == embedding_dim, "Embedding: output last dim must match embedding dim.");
    // Out 的元素总数必须等于 indices数量 * EmbeddingDim
    ASSERT(out->numel() == num_indices * embedding_dim, "Embedding: output size mismatch.");

    // 5. 调度执行
    
    // Always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                              out->dtype(), index->dtype(), 
                              num_indices, embedding_dim, vocab_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                              out->dtype(), index->dtype(), 
                              num_indices, embedding_dim, vocab_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
