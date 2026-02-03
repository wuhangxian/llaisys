#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>


// 模板函数：实际的计算逻辑
template <typename T>
void argmax_impl(int64_t* max_idx, T* max_val, const T* vals, size_t numel) {
    if (numel == 0) return;

    // 初始化：假设第一个元素是最大的
    int64_t best_idx = 0;
    T best_val = vals[0];

    // 遍历剩余元素
    for (size_t i = 1; i < numel; ++i) {
        T curr_val = vals[i];

        // 处理 fp16/bf16 的比较：先转成 float 再比，保证精度和兼容性
        // 如果是普通 float/int，constexpr 会优化掉这部分分支
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float v1 = llaisys::utils::cast<float>(curr_val);
            float v2 = llaisys::utils::cast<float>(best_val);
            if (v1 > v2) {
                best_val = curr_val;
                best_idx = i;
            }
        } else {
            // 标准类型的直接比较
            if (curr_val > best_val) {
                best_val = curr_val;
                best_idx = i;
            }
        }
    }

    // 将结果写回输出指针
    *max_idx = best_idx;
    *max_val = best_val;
}

namespace llaisys::ops::cpu {

void argmax(std::byte* max_idx_ptr, std::byte* max_val_ptr, const std::byte* vals_ptr, llaisysDataType_t dtype, size_t numel) {
    // 假设 max_idx 总是 int64 类型
    auto* idx_out = reinterpret_cast<int64_t*>(max_idx_ptr);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl<float>(
            idx_out, 
            reinterpret_cast<float*>(max_val_ptr), 
            reinterpret_cast<const float*>(vals_ptr), 
            numel);
            
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<llaisys::bf16_t>(
            idx_out,
            reinterpret_cast<llaisys::bf16_t*>(max_val_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(vals_ptr),
            numel);

    case LLAISYS_DTYPE_F16:
        return argmax_impl<llaisys::fp16_t>(
            idx_out,
            reinterpret_cast<llaisys::fp16_t*>(max_val_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(vals_ptr),
            numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu