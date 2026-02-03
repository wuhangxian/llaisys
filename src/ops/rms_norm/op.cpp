#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rmsnorm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
// 1. Check Device Consistency
    CHECK_SAME_DEVICE(out, in, weight);

    // 2. Check Contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMSNorm: inputs/weight/output must be contiguous.");

    // 3. Check Dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // 4. Check Shapes
    // Input/Output: [M, d], Weight: [d]
    ASSERT(in->ndim() == 2, "RMSNorm: Input must be 2D");
    ASSERT(out->ndim() == 2, "RMSNorm: Output must be 2D");
    ASSERT(weight->ndim() == 1, "RMSNorm: Weight must be 1D");

    size_t M = in->shape()[0];
    size_t d = in->shape()[1];

    ASSERT(out->shape()[0] == M, "RMSNorm: Output batch dim must match input.");
    ASSERT(out->shape()[1] == d, "RMSNorm: Output feature dim must match input.");
    ASSERT(weight->shape()[0] == d, "RMSNorm: Weight dim must match input feature dim.");

    // 5. Dispatch
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), M, d, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), M, d, eps);
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
