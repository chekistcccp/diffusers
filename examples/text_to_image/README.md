# text_to_image — 蛋白质信息驱动的鸟类图像生成

本项目通过蛋白质组信息（proteome）生成对应物种的鸟类图像，采用两阶段训练架构。

## 文件说明

| 文件 | 用途 | 状态 |
|------|------|------|
| `train_protein_clip_alignment.py` | **阶段一：对齐训练**。单独训练 ProteinCLIPAdapter，使其输出对齐到 CLIP text encoder 条件空间。三个损失：特征回归（描述性文本 CLIP 编码为目标）、对比学习（CLIP vision）、分类保持 | ✅ 新增，当前使用 |
| `train_text_to_image_lora.py` | **阶段二：Diffusion 训练**。冻结对齐好的 adapter，训练 SD UNet LoRA。使用 ProteinCLIPAdapter 替代原始 speciesModel | ✅ 已修改，当前使用 |
| `inference2.py` | **推理脚本**。加载 adapter 权重 + LoRA 权重，使用 ProteinCLIPAdapter 编码蛋白条件，生成物种图像。guidance_scale 默认 1.0（不使用 CFG） | ✅ 已修改，当前使用 |
| `train_text_to_image.py` | 原始 SD 全参数训练脚本（diffusers 官方） | 未修改，未使用 |
| `train_text_to_image_lora_sdxl.py` | SDXL LoRA 训练脚本（diffusers 官方） | 未修改，未使用 |
| `train_text_to_image_sdxl.py` | SDXL 全参数训练脚本（diffusers 官方） | 未修改，未使用 |
| `train_text_to_image_flax.py` | Flax/JAX 版训练脚本（diffusers 官方） | 未修改，未使用 |
| `train_text_to_image_lora-n1.py` | 早期 LoRA 训练修改版 | ❌ 旧版，仅供参考 |
| `test_text_to_image.py` | 测试脚本 | 未修改 |
| `test_text_to_image_lora.py` | LoRA 测试脚本 | 未修改 |

## 子文件夹

| 文件夹 | 内容 |
|--------|------|
| `pretrained_encoder/` | 编码器模块（ProteinCLIPAdapter、ProtTokenizer、ProtBERT Embedding 等），详见该文件夹说明 |
| `dataprocess/` | 数据处理工具（蛋白采样、图像预处理、描述生成、格式转换等），详见该文件夹说明 |

## 两阶段训练流程

```
阶段一：对齐训练
  蛋白序列 → ProteinCLIPAdapter → protein_hidden_states
                                        ↓
                        MSE vs CLIP_text(描述文本) + InfoNCE vs CLIP_vision(图像) + CrossEntropy(物种分类)
                                        ↓
                        保存 protein_clip_adapter.pt

阶段二：Diffusion 训练
  加载 protein_clip_adapter.pt → ProteinCLIPAdapter (冻结adapter可训练组件)
                                        ↓
                            encoder_hidden_states (77, 768)
                                        ↓
                            SD UNet + LoRA (可训练)
                                        ↓
                            保存 LoRA 权重

推理
  加载 adapter + LoRA → 蛋白序列编码 → Stable Diffusion 生成图像
```

## 关键修改要点

- **不使用 null condition / CFG**：所有代码中无 null_condition_embedding、无 forward_null_condition、无 CFG dropout
- **NPU 兼容**：`get_device()` 自动检测 NPU > CUDA > CPU，`empty_cache()` 兼容 NPU/CUDA
- **DP2 已修复**：subsystem_genes 在蛋白采样中优先保留，不再被覆盖
