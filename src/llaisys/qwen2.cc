#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "llaisys/ops.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>


struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;      // 保存模型元数据
    LlaisysQwen2Weights weights; // 保存权重指针


};


extern "C" {


    static llaisysTensor_t create_tensor_wrapper(
        const std::vector<size_t>& shape, 
        llaisysDataType_t dtype, 
        llaisysDeviceType_t device, 
        int device_id
    ) {
        return tensorCreate(const_cast<size_t*>(shape.data()), shape.size(), dtype, device, device_id);
    }

    // --- Helper to Create an Array of Tensors (for layers) ---
    static llaisysTensor_t* create_layer_tensors(
        size_t nlayer, 
        const std::vector<size_t>& shape, 
        llaisysDataType_t dtype, 
        llaisysDeviceType_t device, 
        int device_id
    ) {
        auto ptr = new llaisysTensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            ptr[i] = create_tensor_wrapper(shape, dtype, device, device_id);
        }
        return ptr;
    }

    // --- Helper to Destroy an Array of Tensors ---
    static void destroy_layer_tensors(llaisysTensor_t* ptr, size_t nlayer) {
        if (!ptr) return;
        for (size_t i = 0; i < nlayer; ++i) {
            tensorDestroy(ptr[i]);
        }
        delete[] ptr;
    }



    // --- 创建模型 ---
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, 
        llaisysDeviceType_t device, 
        int *device_ids, 
        int ndevice
    ) {
        printf("[C++ Backend] Creating Qwen2 Model...\n");
        printf("  - Layers: %zu\n", meta->nlayer);
        printf("  - Hidden: %zu\n", meta->hs);
        fflush(stdout);

        
        // 1. 申请模型结构体内存
        auto model = new LlaisysQwen2Model();
        
        if (!model) return nullptr;

        model->meta = *meta;
        LlaisysQwen2Weights& w = model->weights;

        // Extract params for readability
        size_t nlayer = meta->nlayer;
        llaisysDataType_t dtype = meta->dtype;
        int main_device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

        // Dimensions (cast to size_t for shape vectors)
        size_t hidden = meta->hs;
        size_t vocab = meta->voc;
        size_t intermediate = meta->di;
        // GQA: kv_dim = n_kv_heads * head_dim
        size_t kv_dim = meta->nkvh * meta->dh; 

        // ------------------------------------------------
        // 1. Global Tensors (Single Tensors)
        // ------------------------------------------------
        
        // Token Embeddings: [vocab, hidden]
        w.in_embed = create_tensor_wrapper({vocab, hidden}, dtype, device, main_device_id);
        
        // Output Embeddings (LM Head): [vocab, hidden]
        w.out_embed = create_tensor_wrapper({vocab, hidden}, dtype, device, main_device_id);
        
        // Final Norm: [hidden]
        w.out_norm_w = create_tensor_wrapper({hidden}, dtype, device, main_device_id);

        // ------------------------------------------------
        // 2. Layer Tensors (Array of Tensors)
        // ------------------------------------------------
        // Note: PyTorch linear weights are stored as [out_features, in_features]
        // But for calculation, we might view them differently. 
        // Here we stick to [Out, In] to match Safetensors layout directly.

        // Attention Block
        // Input Norm
        w.attn_norm_w = create_layer_tensors(nlayer, {hidden}, dtype, device, main_device_id);

        // Q: [hidden, hidden] + Bias
        w.attn_q_w = create_layer_tensors(nlayer, {hidden, hidden}, dtype, device, main_device_id);
        w.attn_q_b = create_layer_tensors(nlayer, {hidden}, dtype, device, main_device_id);

        // K: [kv_dim, hidden] + Bias
        w.attn_k_w = create_layer_tensors(nlayer, {kv_dim, hidden}, dtype, device, main_device_id);
        w.attn_k_b = create_layer_tensors(nlayer, {kv_dim}, dtype, device, main_device_id);

        // V: [kv_dim, hidden] + Bias
        w.attn_v_w = create_layer_tensors(nlayer, {kv_dim, hidden}, dtype, device, main_device_id);
        w.attn_v_b = create_layer_tensors(nlayer, {kv_dim}, dtype, device, main_device_id);

        // Output: [hidden, hidden] (Usually no bias for Qwen2 o_proj)
        w.attn_o_w = create_layer_tensors(nlayer, {hidden, hidden}, dtype, device, main_device_id);

        // MLP Block
        // Post Attention Norm
        w.mlp_norm_w = create_layer_tensors(nlayer, {hidden}, dtype, device, main_device_id);

        // Gate: [intermediate, hidden]
        w.mlp_gate_w = create_layer_tensors(nlayer, {intermediate, hidden}, dtype, device, main_device_id);
        
        // Up: [intermediate, hidden]
        w.mlp_up_w = create_layer_tensors(nlayer, {intermediate, hidden}, dtype, device, main_device_id);
        
        // Down: [hidden, intermediate]
        w.mlp_down_w = create_layer_tensors(nlayer, {hidden, intermediate}, dtype, device, main_device_id);

        fprintf(stderr, "[C++ Backend] Allocated tensor objects for %zu layers.\n", nlayer);
        return model;
    }




    // --- 销毁模型 ---
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;

        fprintf(stderr, "[C++ Backend] Destroying Qwen2 Model...\n");

        LlaisysQwen2Weights& w = model->weights;
        size_t nlayer = model->meta.nlayer;

        // =========================================================
        // 1. 释放层级权重 (Layer Tensors)
        // =========================================================
        // Attention
        destroy_layer_tensors(w.attn_norm_w, nlayer);
        
        destroy_layer_tensors(w.attn_q_w, nlayer);
        destroy_layer_tensors(w.attn_q_b, nlayer);
        destroy_layer_tensors(w.attn_k_w, nlayer);
        destroy_layer_tensors(w.attn_k_b, nlayer);
        destroy_layer_tensors(w.attn_v_w, nlayer);
        destroy_layer_tensors(w.attn_v_b, nlayer);
        destroy_layer_tensors(w.attn_o_w, nlayer);

        // MLP
        destroy_layer_tensors(w.mlp_norm_w, nlayer);
        destroy_layer_tensors(w.mlp_gate_w, nlayer);
        destroy_layer_tensors(w.mlp_up_w, nlayer);
        destroy_layer_tensors(w.mlp_down_w, nlayer);

        // =========================================================
        // 2. 释放全局权重 (Global Tensors)
        // =========================================================
        
        // 注意：处理权重共享的情况 (Weight Tying)
        // 如果 out_embed 和 in_embed 指向同一个地址，只需要销毁一次
        if (w.out_embed && w.out_embed != w.in_embed) {
            tensorDestroy(w.out_embed);
        }
        w.out_embed = nullptr; // 避免悬空指针

        if (w.in_embed) {
            tensorDestroy(w.in_embed);
            w.in_embed = nullptr;
        }

        if (w.out_norm_w) {
            tensorDestroy(w.out_norm_w);
            w.out_norm_w = nullptr;
        }

        // =========================================================
        // 3. 释放模型结构体本身
        // =========================================================
        delete model;
        
        fprintf(stderr, "[C++ Backend] Model destroyed successfully.\n");
    }




    // --- 获取权重接口 ---
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if (!model) return nullptr;
        return &model->weights;
    }




    // --- 推理接口 (Stub) ---
    // __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    //     printf("[C++ Backend] Inference triggered on %zu tokens.\n", ntoken);
        
        
    //     return 0;
    // }
    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model * model, 
        int64_t * token_ids, 
        size_t ntoken,
        float temperature,
        float top_p,
        int top_k
    ) {
        if (!model || !token_ids || ntoken == 0) return -1;
        // fprintf(stderr, "[C++ Backend] Inference triggered on %zu tokens.\n", ntoken);

        // // 获取配置和权重引用
        const auto& meta = model->meta;
        // const auto& w = model->weights;
    
        // 
        // 流程图解：TokenIDs -> Embed -> [Norm -> Attn -> Norm -> MLP] * N -> Norm -> Head -> NextToken

        // =================================================================
        // 1. Embedding Layer
        // =================================================================



        for (size_t i = 0; i < meta.nlayer; ++i) {
            
        }

        return 0;     
    }
} // extern "C"