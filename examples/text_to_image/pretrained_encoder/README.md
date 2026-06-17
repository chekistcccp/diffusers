# pretrained_encoder — 编码器模块

本文件夹包含蛋白质序列编码器的所有组件，核心功能是将蛋白质序列编码为 CLIP 兼容的条件表征。

## 文件说明

| 文件 | 用途 | 状态 |
|------|------|------|
| `protein_clip_adapter.py` | **核心模块**。ProteinCLIPAdapter 将蛋白序列编码为 CLIP 兼容的 (77, 768) 条件表征；AttentionCompressionFixed 将 ProtBERT 输出从 (6000, 1024) 压缩为 (75, 768)；包含 NPU 兼容的 `get_device()` 和 `empty_cache()` 工具函数 | ✅ 新增，当前使用 |
| `tokenizer.py` | ProtTokenizer，蛋白质序列分词器。将含 `[GRP]` 分隔符的蛋白序列转为 token_ids + attention_mask | 未修改，直接复用 |
| `model.py` | EmbeddingFromPretrained，ProtBERT 预训练 embedding 查表层。加载预计算的蛋白 embedding 文件 | 未修改，直接复用 |
| `config.yml` | 编码器配置文件（词汇表路径、embedding 维度等） | 未修改 |
| `attentionCompression.py` | 旧版压缩模块，存在无位置编码、初始化不稳定、mask 填充值不当等问题 | ❌ 已被 AttentionCompressionFixed 替代，保留仅供参考 |
| `dataset.py` | 旧版数据集类，对齐训练改用 parquet 格式后不再使用 | ❌ 不再使用 |
| `trainer.py` | 旧版训练器，已被 `train_protein_clip_alignment.py` 替代 | ❌ 不再使用 |
| `inference.py` | 旧版推理脚本，已被 `inference2.py` 替代 | ❌ 不再使用 |
| `run.py` | 旧版运行入口 | ❌ 不再使用 |

## 数据流

```
蛋白序列 → ProtTokenizer → EmbeddingFromPretrained → AttentionCompressionFixed
                                                              ↓ (75, 768)
                                                    + SOS + EOS → (77, 768)
                                                    + Position Embedding
                                                              ↓
                                                    CLIP Text Encoder (冻结)
                                                              ↓
                                              (last_hidden_state, pooled_output)
```

## 关键设计

- **绕过 CLIP token embedding**：直接将 `inputs_embeds` 输入 CLIP transformer，跳过其 token embedding 层
- **SOS/EOS 可学习边界 token**：75 compressed + 1 SOS + 1 EOS = 77，与 CLIP model_max_length 一致
- **位置编码从 CLIP 初始化**：复制 CLIP position_embedding 权重，但设为可训练
- **不使用 null condition / CFG**：无 null_condition_embedding，无 forward_null_condition
