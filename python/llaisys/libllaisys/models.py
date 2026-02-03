
from ctypes import (
    Structure, 
    POINTER, 
    c_size_t, 
    c_float, 
    c_int64, 
    c_int, 
    c_void_p, 
    byref
)
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t

# =========================================================================
# 1. 结构体定义 (Struct Definitions)
# =========================================================================

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t), # llaisysDataType_t dtype;
        ("nlayer", c_size_t),         # size_t nlayer;
        ("hs", c_size_t),             # size_t hs;
        ("nh", c_size_t),             # size_t nh;
        ("nkvh", c_size_t),           # size_t nkvh;
        ("dh", c_size_t),             # size_t dh;
        ("di", c_size_t),             # size_t di;
        ("maxseq", c_size_t),         # size_t maxseq;
        ("voc", c_size_t),            # size_t voc;
        ("epsilon", c_float),         # float epsilon;
        ("theta", c_float),           # float theta;
        ("end_token", c_int64),       # int64_t end_token;
    ]



class LlaisysQwen2Weights(Structure):
    _fields_ = [
        # 1. 非层级权重 (Global Weights)
        ("in_embed", llaisysTensor_t),    # llaisysTensor_t in_embed;
        ("out_embed", llaisysTensor_t),   # llaisysTensor_t out_embed;
        ("out_norm_w", llaisysTensor_t),  # llaisysTensor_t out_norm_w;

        # 2. 层级权重 (Layer Weights - Arrays of Tensors)
        ("attn_norm_w", POINTER(llaisysTensor_t)), # llaisysTensor_t *attn_norm_w;
        ("attn_q_w", POINTER(llaisysTensor_t)),    # llaisysTensor_t *attn_q_w;
        ("attn_q_b", POINTER(llaisysTensor_t)),    # llaisysTensor_t *attn_q_b;
        ("attn_k_w", POINTER(llaisysTensor_t)),    # ...
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

# 定义模型句柄类型 (Opaque Pointer)
LlaisysQwen2Model_t = c_void_p


# =========================================================================
# 2. 函数加载器 (Function Loader)
# =========================================================================

llaisysQwen2Model_t = c_void_p


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t


    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None


    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)


    # lib.llaisysQwen2ModelInfer.argtypes = [
    #         llaisysQwen2Model_t,  # model 指针
    #         POINTER(c_int64),     # token_ids 数组 (int64_t*)
    #         c_size_t              # ntoken 长度 (size_t)
    #     ]
    # lib.llaisysQwen2ModelInfer.restype = c_int64  # 返回生成的 token id
    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,  # 1. model 指针
        POINTER(c_int64),     # 2. token_ids 数组
        c_size_t,             # 3. ntoken 长度
        c_float,              # 4. temperature (新增)
        c_float,              # 5. top_p (新增)
        c_int                 # 6. top_k (新增)
    ]
    
    lib.llaisysQwen2ModelInfer.restype = c_int64


    # lib.llaisysQwen2ModelKCache.argtypes = [llaisysQwen2Model_t, c_size_t]
    # lib.llaisysQwen2ModelKCache.restype = llaisysTensor_t

    # lib.llaisysQwen2ModelVCache.argtypes = [llaisysQwen2Model_t, c_size_t]
    # lib.llaisysQwen2ModelVCache.restype = llaisysTensor_t











