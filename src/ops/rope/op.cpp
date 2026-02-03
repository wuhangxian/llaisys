#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. Check Device Consistency
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // 2. Check Contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: inputs/output/pos_ids must be contiguous.");

    // 3. Check Dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 4. Check Shapes
    // Input/Output: [SeqLen, Heads, Dim]
    ASSERT(in->ndim() == 3, "RoPE: Input must be 3D [seqlen, nhead, d]");
    ASSERT(out->ndim() == 3, "RoPE: Output must be 3D [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen]");

    size_t L = in->shape()[0]; // SeqLen
    size_t H = in->shape()[1]; // Heads
    size_t D = in->shape()[2]; // Dim

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(pos_ids->shape()[0] == L, "RoPE: pos_ids length must match sequence length.");
    ASSERT(D % 2 == 0, "RoPE: Head dimension must be even.");

    // 5. Dispatch
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(), 
            in->data(), 
            reinterpret_cast<const int64_t*>(pos_ids->data()), 
            out->dtype(), 
            L, H, D, 
            theta
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), reinterpret_cast<const int64_t*>(pos_ids->data()), out->dtype(), L, H, D, theta);
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