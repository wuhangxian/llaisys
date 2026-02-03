from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import safetensors
import json
from ctypes import byref, c_int, c_size_t, c_float, c_int64, c_uint32, c_void_p, memmove

import re

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    llaisysDataType_t,
    llaisysDeviceType_t,
)

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        model_path = Path(model_path)
        config_path = model_path / "config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        torch_dtype = str(cfg.get("torch_dtype", "bfloat16")).lower()
        if "float32" in torch_dtype or torch_dtype in {"fp32", "f32"}:
            dtype = DataType.F32
        elif "float16" in torch_dtype or torch_dtype in {"fp16", "f16"}:
            dtype = DataType.F16
        else:
            dtype = DataType.BF16
        # 统一用 torch 读取 bfloat16，并降级为 float16，避免 numpy bfloat16 兼容问题
        use_torch_loader = False
        if dtype == DataType.BF16:
            dtype = DataType.F16
            use_torch_loader = True
        # 解析模型参数
        nlayer = int(cfg.get("num_hidden_layers", 0))
        hs = int(cfg.get("hidden_size", 0))
        nh = int(cfg.get("num_attention_heads", 0))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg.get("intermediate_size", 0))
        maxseq = int(cfg.get("max_position_embeddings", 0))
        voc = int(cfg.get("vocab_size", 0))
        epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        eos = cfg.get("eos_token_id", -1)
        # 解析结束token
        if isinstance(eos, list):
            end_token = int(eos[0]) if eos else -1
        else:
            end_token = int(eos)
        # 解析head_dim
        dh = int(cfg.get("head_dim", hs // nh if nh else 0))
        

        # debug 信息
        print(f"\n{'='*20} Qwen2 Model Config Info {'='*20}")
        print(f"1.  Model Path:       {model_path}")
        print(f"2.  Compute Dtype:    {dtype} (Original: {torch_dtype})")
        print(f"3.  Use Torch Loader: {use_torch_loader}")
        print(f"4.  Layers (nlayer):  {nlayer}")
        print(f"5.  Hidden Size (hs): {hs}")
        print(f"6.  Attn Heads (nh):  {nh}")
        print(f"7.  KV Heads (nkvh):  {nkvh}  [{'GQA' if nkvh != nh else 'MHA'}]")
        print(f"8.  Head Dim (dh):    {dh}")
        print(f"9.  FFN Dim (di):     {di}")
        print(f"10. Vocab Size (voc): {voc}")
        print(f"11. Max Seq Len:      {maxseq}")
        print(f"12. RoPE Theta:       {theta}")
        print(f"13. Norm Epsilon:     {epsilon}")
        print(f"14. EOS Token ID:     {end_token}")
        print(f"{'='*60}\n")

        model_meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
            c_size_t(nlayer),
            c_size_t(hs),
            c_size_t(nh),
            c_size_t(nkvh),
            c_size_t(dh),
            c_size_t(di),
            c_size_t(maxseq),
            c_size_t(voc),
            c_float(epsilon),
            c_float(theta),
            c_int64(end_token),
        )

        print("python call llaisysQwen2ModelCreate")
        device_ids = (c_int * 1)(0)

        # 这个函数用于为tensor分配内存
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(model_meta),
            llaisysDeviceType_t(device),
            device_ids,
            1,
        )

        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")

        self._model_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._meta = model_meta

        w_struct = self._model_weights.contents

        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(".safetensors file not exsist")
        
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = list(model_path.glob("*.bin"))

        print(f"Loading weights from {len(weight_files)} files...")
        layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")

        for wf in weight_files:
            # 始终使用 pt 框架打开，这样可以直接获得 Tensor 对象
            with safetensors.safe_open(wf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # 获取 Tensor (PyTorch Tensor)
                    tensor = f.get_tensor(key)
                    
                    # === 关键逻辑：处理数据类型转换 ===
                    if use_torch_loader:
                        # 如果原始是 BF16，这里必须转为 F16，否则 C++ 端全是乱码
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float16)
                    elif dtype == DataType.F32 and tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    
                    # 获取源数据指针和大小
                    # 注意：必须确保 tensor 是连续的，否则 data_ptr 可能不正确
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        
                    src_ptr = tensor.data_ptr()
                    src_size = tensor.numel() * tensor.element_size()

                    # === 名称映射 (Map Name -> C Pointer) ===
                    dest_tensor = None
                    
                    if key == "model.embed_tokens.weight":
                        dest_tensor = w_struct.in_embed
                    elif key == "lm_head.weight":
                        dest_tensor = w_struct.out_embed
                    elif key == "model.norm.weight":
                        dest_tensor = w_struct.out_norm_w
                    else:
                        match = layer_pattern.search(key)
                        if match:
                            layer_idx = int(match.group(1))
                            suffix = match.group(2)
                            
                            # 根据后缀映射到对应的指针数组
                            if "input_layernorm.weight" in suffix:
                                dest_tensor = w_struct.attn_norm_w[layer_idx]
                            
                            elif "self_attn.q_proj.weight" in suffix:
                                dest_tensor = w_struct.attn_q_w[layer_idx]
                            elif "self_attn.q_proj.bias" in suffix:
                                dest_tensor = w_struct.attn_q_b[layer_idx]
                                
                            elif "self_attn.k_proj.weight" in suffix:
                                dest_tensor = w_struct.attn_k_w[layer_idx]
                            elif "self_attn.k_proj.bias" in suffix:
                                dest_tensor = w_struct.attn_k_b[layer_idx]
                                
                            elif "self_attn.v_proj.weight" in suffix:
                                dest_tensor = w_struct.attn_v_w[layer_idx]
                            elif "self_attn.v_proj.bias" in suffix:
                                dest_tensor = w_struct.attn_v_b[layer_idx]
                                
                            elif "self_attn.o_proj.weight" in suffix:
                                dest_tensor = w_struct.attn_o_w[layer_idx]
                                
                            elif "post_attention_layernorm.weight" in suffix:
                                dest_tensor = w_struct.mlp_norm_w[layer_idx]
                                
                            elif "mlp.gate_proj.weight" in suffix:
                                dest_tensor = w_struct.mlp_gate_w[layer_idx]
                            elif "mlp.up_proj.weight" in suffix:
                                dest_tensor = w_struct.mlp_up_w[layer_idx]
                            elif "mlp.down_proj.weight" in suffix:
                                dest_tensor = w_struct.mlp_down_w[layer_idx]

                    # === 执行内存拷贝 ===
                    if dest_tensor is not None and dest_tensor != 0:
                        # 这里通过 ctypes.memmove 将 Python(Torch) 内存块复制到 C++ 内存块
                        # dest_tensor类型为 llaisysTensor_t
                        LIB_LLAISYS.tensorLoad(dest_tensor, c_void_p(src_ptr))

        print("Model initialization complete.")


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # TODO: Implement generate function
        current_ids = list(inputs)

        # 3. 循环生成
        for _ in range(max_new_tokens):
            seq_len = len(current_ids)
            c_tokens = (c_int64 * seq_len)(*current_ids)

            # 调用 C++ 推理
            # 注意：这里每次都把完整的 current_ids 传进去
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, c_tokens, c_size_t(seq_len),
                c_float(temperature), c_float(top_p), c_int(top_k)
            )

            # 结束条件
            if next_token == self._meta.end_token:
                break
            
            current_ids.append(next_token)
            # print(f"Generated: {next_token}") # Debug

        return current_ids
