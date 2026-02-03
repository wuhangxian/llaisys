#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
// 1. Check Device Consistency
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    // 2. Check Input Contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: inputs/weight/output must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // 3. Check Dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // 4. Check Shapes for MatMul: Y(M, N) = X(M, K) * W^T(K, N) + b(N)
    // weight shape is [N, K] physically.
    ASSERT(in->ndim() == 2, "Linear: Input must be 2D");
    ASSERT(weight->ndim() == 2, "Linear: Weight must be 2D");
    ASSERT(out->ndim() == 2, "Linear: Output must be 2D");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0]; // Weight is [N, K]

    ASSERT(weight->shape()[1] == K, "Linear: Weight feature dim must match input feature dim (K).");
    ASSERT(out->shape()[0] == M, "Linear: Output batch dim must match input batch dim (M).");
    ASSERT(out->shape()[1] == N, "Linear: Output feature dim must match weight output dim (N).");

    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: Bias must be 1D");
        ASSERT(bias->shape()[0] == N, "Linear: Bias dim must match output feature dim (N).");
    }

    // 5. Dispatch
    // Always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, // Handle optional bias
            out->dtype(), 
            M, N, K
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), M, N, K);
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
