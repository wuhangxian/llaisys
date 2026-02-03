#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // TO_BE_IMPLEMENTED();
    // 1. 检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    // 2. 检查数据类型
    // max_val 应该和 vals 类型一致
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // 3. 检查维度 (根据你的题目要求，暂时只支持 1D)
    ASSERT(vals->ndim() == 1, "argmax: currently only supports 1D input.");
    ASSERT(max_val->numel() == 1, "argmax: max_val must be a scalar (1-element tensor).");
    ASSERT(max_idx->numel() == 1, "argmax: max_idx must be a scalar (1-element tensor).");
    
    // 4. 检查连续性
    ASSERT(vals->isContiguous(), "argmax: input must be contiguous.");
    ASSERT(max_val->isContiguous(), "argmax: max_val must be contiguous.");
    ASSERT(max_idx->isContiguous(), "argmax: max_idx must be contiguous.");

    // 5. 设备分发
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用 CPU 内核
        return cpu::argmax(
            max_idx->data(), // 输出索引的指针
            max_val->data(), // 输出最大值的指针
            vals->data(),    // 输入数据的指针
            vals->dtype(),   // 数据类型
            vals->numel()    // 元素个数
        );
    }

    // 设置设备上下文 (为后续 GPU 支持做准备)
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
