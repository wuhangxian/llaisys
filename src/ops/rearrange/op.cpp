#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

#include <cstring> // for std::memcpy

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 1. Check Device Consistency
    CHECK_SAME_DEVICE(out, in);

    // 2. Check Dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 3. Check Shape
    // Rearrange requires the logical shape to be exactly the same.
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // 4. Optimization: If both are contiguous, just memcpy
    if (out->isContiguous() && in->isContiguous()) {
        size_t bytes = out->numel() * out->elementSize();
        // Assuming tensor_t->data() returns void* or byte*
        std::memcpy(out->data(), in->data(), bytes);
        return;
    }

    // 5. Prepare Metadata for generic stride copy
    size_t ndim = out->ndim();
    const auto& shape = out->shape();
    const auto& out_strides = out->strides();
    const auto& in_strides = in->strides();

    // 6. Dispatch
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(
            out->data(), in->data(),
            out->dtype(),
            shape,
            out_strides,
            in_strides,
            ndim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(
            out->data(), in->data(),
            out->dtype(),
            shape,
            out_strides,
            in_strides,
            ndim
        );
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