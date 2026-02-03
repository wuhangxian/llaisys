#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. Check Device Consistency
    CHECK_SAME_DEVICE(out, gate, up);

    // 2. Check Contiguity
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

    // 3. Check Dtype
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 4. Check Shapes
    // All inputs must have the same shape
    CHECK_SAME_SHAPE(out->shape(), gate->shape());
    CHECK_SAME_SHAPE(out->shape(), up->shape());

    // 5. Dispatch
    size_t numel = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
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