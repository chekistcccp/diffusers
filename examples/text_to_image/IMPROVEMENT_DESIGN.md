# 蛋白信息→视觉描述文本→图像生成：改进设计方案

> 版本: v1.0
> 日期: 2026-06-09
> 状态: 设计阶段

---

## 1. 背景与动机

### 1.1 当前方案的问题

当前 `ProteinCLIPAdapter` 方案将蛋白ID embedding 压缩后注入 CLIP text encoder，再由 Stable Diffusion UNet 的 cross-attention 消费。该方案存在以下根本性问题：

| 问题编号 | 问题描述 | 严重程度 |
|----------|----------|----------|
| P0-1 | `guidance_scale=1.0` 导致无 CFG 引导，图像生成无意义 | 致命 |
| P0-2 | `ProtTokenizer.model_max_length=6000` 与 CLIP 的 77 不匹配 | 致命 |
| P0-3 | 推理时 unconditional embedding 路径与 conditional 不一致 | 致命 |
| P1-1 | Diffusion 训练冻结所有 adapter 参数，无法端到端优化 | 严重 |
| P1-2 | Contrastive loss 温度参数过大 (100/0.07≈1428.6) | 严重 |
| P1-3 | CompressionModule (75×768) 完全为 CLIP 量身定制，不可复用 | 严重 |
| P2-1 | Padding 位置的异常输出可能影响 EOS token 表示质量 | 中等 |

### 1.2 改进思路

**核心转变**：从"蛋白embedding→CLIP空间→UNet"的端到端方案，转为**两阶段解耦方案**：

```
阶段1: 蛋白ID列表 → [蛋白→文本模型] → 视觉描述文本
阶段2: 视觉描述文本 → [标准扩散模型+LoRA] → 图像
```

**优势**：
- 两个阶段各自独立，训练和调试更简单
- 中间产物（文本）可读可检查
- 阶段2可复用成熟的 SD/SDXL/FLUX 生态，无需修改扩散模型
- 数据更易获取：蛋白-文本对远比蛋白-图像对丰富

---

## 2. 数据现状与数据管道设计

### 2.1 现有数据资产

| 数据 | 格式 | 内容 | 规模 |
|------|------|------|------|
| 蛋白ID-图像对应 | `.out` / `.validate` (tab分隔) | `图像路径\t蛋白ID列表` | ~124K 训练对 |
| 图像-描述对应 | `16-birds-descriptions.csv` | `图像路径,描述文本` | ~数千条 |
| 蛋白预训练embedding | `embeddings_prot_bert_bfd1/` | `.npy` + `.idx` 文件 | 16种鸟 |
| 物种-蛋白子系统 | `genes_by_function/` | 按功能分组的蛋白列表 | 8个功能组 |
| 图像-文本HF数据集 | `16birds-text-image-data/` | Arrow格式 (image, description) | 124K 训练 / 3个验证分片 |

### 2.2 数据关联关系

```
数据1 (.out文件):  图像路径 ←→ 蛋白ID列表
数据2 (CSV文件):   图像路径 ←→ 视觉描述文本
数据3 (embedding): 蛋白ID   ←→ 1024维向量

通过"图像路径"作为join key:
  蛋白ID列表 ←→ 图像路径 ←→ 视觉描述文本
  ↓
  蛋白ID列表 ←→ 视觉描述文本  (阶段1训练对)
```

### 2.3 数据合并脚本设计

```python
# build_protein_text_pairs.py
# 输入: .out文件 + .validate文件 + 16-birds-descriptions.csv
# 输出: protein_text_pairs.jsonl
# 格式: {"protein_ids": "tr|A0A7L3F1E2|...,tr|A0A7L3FWF2|...", "description": "A bird with..."}

import csv
import json
from pathlib import Path
from collections import defaultdict

def load_protein_image_pairs(out_files):
    """加载 蛋白ID列表↔图像路径 对"""
    pairs = {}  # image_path -> protein_ids
    for f in out_files:
        with open(f, 'r') as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs[parts[0]] = parts[1]
    return pairs

def load_image_descriptions(csv_path):
    """加载 图像路径↔描述文本 对"""
    descs = {}  # image_path -> description
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                descs[row[0].strip()] = row[1].strip()
    return descs

def build_pairs(protein_image, image_desc):
    """通过图像路径关联，构建 蛋白ID列表↔描述文本 对"""
    results = []
    for img_path, protein_ids in protein_image.items():
        if img_path in image_desc:
            results.append({
                "protein_ids": protein_ids,
                "description": image_desc[img_path],
            })
    return results

# 注意: 同一物种的图像共享同一组蛋白ID列表
# 可选: 按物种聚合描述，生成物种级别的统一描述
```

### 2.4 数据增强策略

由于只有16种鸟（16种不同的蛋白ID列表），数据多样性严重不足。需要以下增强：

| 增强方式 | 方法 | 效果 |
|----------|------|------|
| 蛋白ID随机dropout | 以概率 p=0.1 随机移除部分蛋白ID | 增加输入多样性 |
| 蛋白ID随机重排 | 随机打乱蛋白ID顺序 | 消除顺序依赖 |
| 子系统分组采样 | 从8个功能组中随机采样子集 | 增加组合多样性 |
| 描述文本改写 | 用LLM对现有描述进行改写/扩展 | 增加输出多样性 |
| 增加物种数量 | 扩展到100+种鸟（embedding目录已支持） | **最关键的增强** |

---

## 3. 阶段1：蛋白→文本模型架构设计

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    ProteinToTextModel                             │
│                                                                   │
│  输入: 蛋白ID列表 (逗号分隔的蛋白UniProt ID)                        │
│                                                                   │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │ ProtTokenizer│───→│EmbeddingFromPre-  │───→│ ProteinEncoder   │ │
│  │ (保留不变)   │    │ trained (冻结)    │    │ (新设计, 可训练)  │ │
│  │             │    │ 6000×1024        │    │ 6000×1024        │ │
│  └─────────────┘    └──────────────────┘    │ → N×1536         │ │
│                                              └────────┬─────────┘ │
│                                                       │           │
│                                              ┌────────▼─────────┐ │
│                                              │ Qwen2.5-1.5B     │ │
│                                              │ (LoRA微调)       │ │
│                                              │ 输入: [虚拟token] │ │
│                                              │   + [BOS+文本]   │ │
│                                              │ 输出: 描述文本    │ │
│                                              └──────────────────┘ │
│                                                                   │
│  输出: 视觉描述文本 (如 "A bird with outstretched blue wings...")  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 ProteinEncoder（替代 CompressionModule + SOS/EOS + position_embedding + CLIP encoder）

**设计原则**：
- 不再受 CLIP 的 77 token / 768 维限制
- 输出维度匹配 Qwen2.5-1.5B 的 hidden_size (1536)
- 输出 token 数量自由选择
- 使用多层 cross-attention 从蛋白 embedding 中提取信息

```python
class ProteinEncoder(nn.Module):
    """
    将蛋白ID embedding (batch, 6000, 1024) 压缩为
    虚拟token序列 (batch, num_output_tokens, output_dim)

    替代原方案中的:
    - AttentionCompressionFixed (75×768)
    - sos_token / eos_token
    - position_embedding (77, 768)
    - CLIP text encoder (12层transformer)
    """

    def __init__(self,
                 input_dim=1024,          # ProtBert-BFD embedding维度
                 output_dim=1536,         # Qwen2.5-1.5B hidden_size
                 num_input_tokens=6000,   # 最大蛋白ID数量
                 num_output_tokens=64,    # 压缩后的虚拟token数
                 num_heads=8,             # cross-attention头数
                 num_layers=2,            # cross-attention层数
                 ffn_dim_multiplier=4):   # FFN中间层倍数
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_output_tokens = num_output_tokens

        # 可学习的query token (类似BLIP-2 Q-Former)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, input_dim) * 0.02
        )

        # 多层 cross-attention + FFN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Cross-attention: query attend to protein embeddings
                'ln_q': nn.LayerNorm(input_dim),
                'ln_kv': nn.LayerNorm(input_dim),
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                'ln_cross_attn': nn.LayerNorm(input_dim),
                # Self-attention: query attend to each other
                'ln_self_q': nn.LayerNorm(input_dim),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                'ln_self_attn': nn.LayerNorm(input_dim),
                # FFN
                'ln_ffn': nn.LayerNorm(input_dim),
                'ffn': nn.Sequential(
                    nn.Linear(input_dim, input_dim * ffn_dim_multiplier),
                    nn.GELU(),
                    nn.Linear(input_dim * ffn_dim_multiplier, input_dim),
                ),
            }))

        # 输出投影: input_dim → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, protein_embeds, attention_mask=None):
        """
        Args:
            protein_embeds: (batch, num_proteins, 1024) 蛋白embedding
            attention_mask: (batch, num_proteins) 1=有效, 0=padding
        Returns:
            virtual_tokens: (batch, num_output_tokens, 1536)
        """
        batch_size = protein_embeds.shape[0]

        # 初始化query
        query = self.query_tokens.expand(batch_size, -1, -1)

        # 构建key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True=忽略

        # 逐层处理
        for layer in self.layers:
            # 1. Cross-attention: query从蛋白embedding提取信息
            residual = query
            q_norm = layer['ln_q'](query)
            kv_norm = layer['ln_kv'](protein_embeds)
            attn_out, _ = layer['cross_attn'](
                q_norm, kv_norm, kv_norm,
                key_padding_mask=key_padding_mask,
            )
            query = layer['ln_cross_attn'](residual + attn_out)

            # 2. Self-attention: query之间交互
            residual = query
            q_norm = layer['ln_self_q'](query)
            self_out, _ = layer['self_attn'](
                q_norm, q_norm, q_norm,
            )
            query = layer['ln_self_attn'](residual + self_out)

            # 3. FFN
            residual = query
            query = residual + layer['ffn'](layer['ln_ffn'](query))

        # 投影到Qwen维度
        virtual_tokens = self.output_proj(query)
        return virtual_tokens
```

### 3.3 `num_output_tokens` 选择指南

| 值 | 适用场景 | 信息保留 | 计算量 | 推荐度 |
|----|----------|----------|--------|--------|
| 16 | 极简验证 | 低 | 极小 | 快速实验 |
| 32 | 轻量级 | 中等 | 小 | 推荐起点 |
| **64** | **标准配置** | **较好** | **适中** | **推荐** |
| 128 | 信息充分 | 好 | 较大 | 数据量大时 |
| 75 | 沿用旧值 | — | — | 不推荐(无特殊意义) |

**推荐从 64 开始**，理由：
- 16种鸟的蛋白组差异需要足够的token来编码
- 64个虚拟token在Qwen的32768上下文中占比很小
- 与BLIP-2的Q-Former (32~64 tokens) 经验一致

### 3.4 Qwen2.5 解码器的输入构造

```python
def build_decoder_inputs(virtual_tokens, text_ids, text_decoder):
    """
    构造Qwen解码器的输入:
    [蛋白虚拟token] + [文本token embedding]

    Args:
        virtual_tokens: (batch, N, 1536) ProteinEncoder输出
        text_ids: (batch, text_len) 目标文本的token IDs
        text_decoder: Qwen2.5模型

    Returns:
        inputs_embeds: (batch, N + text_len, 1536)
        labels: (batch, N + text_len) -100表示不计算loss的位置
    """
    # 获取文本token的embedding
    text_embeds = text_decoder.get_input_embeddings()(text_ids)

    # 拼接: [虚拟token] + [文本token]
    inputs_embeds = torch.cat([virtual_tokens, text_embeds], dim=1)

    # 构造labels: 虚拟token位置不计算loss
    N = virtual_tokens.shape[1]
    labels = text_ids.clone()
    prefix_labels = torch.full(
        (text_ids.shape[0], N), -100,
        device=text_ids.device, dtype=text_ids.dtype
    )
    labels = torch.cat([prefix_labels, labels], dim=1)

    return inputs_embeds, labels
```

### 3.5 完整模型类

```python
class ProteinToTextModel(nn.Module):
    """
    蛋白ID列表 → 视觉描述文本

    训练目标: 给定蛋白ID列表，自回归生成对应的鸟类视觉描述文本
    """

    def __init__(self,
                 embeddings_dir,
                 vocabulary_path,
                 species_classes_path,
                 text_decoder_name="Qwen/Qwen2.5-1.5B",
                 num_output_tokens=64,
                 max_protein_tokens=6000,
                 protein_embedding_dim=1024,
                 lora_r=16,
                 lora_alpha=32,
                 device=None):
        super().__init__()

        # 1. 蛋白Tokenizer (保留不变)
        self.tokenizer = ProtTokenizer(...)

        # 2. 蛋白Embedding层 (保留不变, 冻结)
        self.protein_embedding = EmbeddingFromPretrained(
            vector_size=protein_embedding_dim,
            embed_dir=embeddings_dir,
            sequence_max_length=max_protein_tokens,
            device=device,
        )
        for param in self.protein_embedding.parameters():
            param.requires_grad = False

        # 3. 蛋白编码器 (新设计)
        # 获取text_decoder的hidden_size
        text_decoder_config = AutoConfig.from_pretrained(text_decoder_name)
        decoder_hidden_size = text_decoder_config.hidden_size

        self.protein_encoder = ProteinEncoder(
            input_dim=protein_embedding_dim,
            output_dim=decoder_hidden_size,
            num_input_tokens=max_protein_tokens,
            num_output_tokens=num_output_tokens,
        )

        # 4. 文本解码器 (Qwen2.5 + LoRA)
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            text_decoder_name,
            torch_dtype=torch.bfloat16,
        )

        # 添加LoRA
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.text_decoder = get_peft_model(self.text_decoder, lora_config)

    def forward(self, protein_ids, protein_attention_mask, text_ids):
        """
        Args:
            protein_ids: (batch, 6000) 蛋白ID token IDs
            protein_attention_mask: (batch, 6000)
            text_ids: (batch, text_len) 目标描述文本token IDs
        """
        # 1. 蛋白embedding
        protein_embeds, _ = self.protein_embedding(protein_ids)

        # 2. 压缩为虚拟token
        virtual_tokens = self.protein_encoder(
            protein_embeds, protein_attention_mask
        )  # (batch, N, decoder_hidden_size)

        # 3. 构造解码器输入
        inputs_embeds, labels = build_decoder_inputs(
            virtual_tokens, text_ids, self.text_decoder
        )

        # 4. 自回归生成
        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )

        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate(self, protein_ids, protein_attention_mask, max_new_tokens=128):
        """推理: 蛋白ID列表 → 描述文本"""
        protein_embeds, _ = self.protein_embedding(protein_ids)
        virtual_tokens = self.protein_encoder(
            protein_embeds, protein_attention_mask
        )

        # 只输入虚拟token，让模型自回归生成
        outputs = self.text_decoder.generate(
            inputs_embeds=virtual_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        return self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 4. 训练策略设计

### 4.1 两阶段训练（参考 Prot2Text-V2）

#### 阶段1: 对比对齐（~5-10 epoch）

**目标**：让 ProteinEncoder 的输出与文本语义空间对齐

**冻结**：
- `EmbeddingFromPretrained` (冻结)
- `Qwen2.5-1.5B` (冻结)

**训练**：
- `ProteinEncoder` (可训练)

**Loss**：对比学习 (InfoNCE)

```python
def contrastive_loss(protein_features, text_features, temperature=0.07):
    """
    protein_features: (batch, dim) - 虚拟token的mean-pooling
    text_features: (batch, dim) - 文本decoder最后一层hidden的mean-pooling
    temperature: 可学习参数, 初始化为 log(1/0.07)
    """
    # L2归一化
    protein_features = F.normalize(protein_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度矩阵
    logits = protein_features @ text_features.T / temperature

    # 对称对比loss
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

**注意**：温度参数使用可学习的 `logit_scale`，初始化为 `log(1/0.07) ≈ 2.66`，而非硬编码的 `100/0.07 ≈ 1428.6`。

#### 阶段2: 生成微调（~20 epoch）

**目标**：给定蛋白ID列表，生成对应的视觉描述文本

**冻结**：
- `EmbeddingFromPretrained` (冻结)
- `ProteinEncoder` (冻结，阶段1已训练好)

**训练**：
- `Qwen2.5-1.5B LoRA` (可训练)

**Loss**：交叉熵（自回归文本生成）

```python
# 标准的自回归生成loss，虚拟token位置mask为-100
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
```

### 4.2 训练超参数

| 参数 | 阶段1 (对齐) | 阶段2 (生成) |
|------|-------------|-------------|
| batch_size | 32 | 16 |
| learning_rate | 1e-4 | 2e-5 (LoRA) |
| optimizer | AdamW | AdamW |
| scheduler | cosine | cosine |
| epochs | 5-10 | 20 |
| warmup_steps | 500 | 200 |
| weight_decay | 0.01 | 0.01 |
| gradient_accumulation | 4 | 4 |
| mixed_precision | bf16 | bf16 |
| num_output_tokens | 64 | 64 |

### 4.3 数据量与扩展计划

| 阶段 | 鸟类物种数 | 训练样本数 | 预期效果 |
|------|-----------|-----------|----------|
| 验证期 | 16 | ~128K (图像级) / 16 (物种级) | 验证pipeline可跑通 |
| 扩展期 | 50-100 | ~500K | 初步泛化能力 |
| 完整期 | 300+ | ~2M | 较强泛化能力 |

**关键**：embedding 目录中已有远超16种鸟的预训练 embedding，扩展物种数量是可行的。

---

## 5. 阶段2：文本→图像（扩散模型）

### 5.1 设计原则

**阶段2 完全不需要修改扩散模型架构**。只需要：

1. 将阶段1生成的视觉描述文本作为 prompt
2. 使用标准 SD/SDXL/FLUX + LoRA 微调生成图像

### 5.2 推理流程

```
输入: 蛋白ID列表
  ↓
ProteinToTextModel.generate()
  ↓
视觉描述文本 (如 "A bird with long blue wings and a sharp yellow beak")
  ↓
标准扩散模型推理 (SD/SDXL/FLUX)
  ↓
生成的鸟类图像
```

### 5.3 扩散模型选择

| 模型 | 优势 | 劣势 | 推荐场景 |
|------|------|------|----------|
| SD 1.5 + LoRA | 轻量, 你已有代码 | 生成质量一般 | 快速验证 |
| SDXL + LoRA | 质量好, 生态成熟 | 需要12GB+ VRAM | **推荐** |
| FLUX.1-s + LoRA | 质量最好 | 需要24GB+ VRAM | 高质量生成 |

### 5.4 LoRA 微调数据

使用现有的 `16birds-text-image-data/` 数据集：
- 图像: 鸟类图像
- 文本: Qwen3.6 生成的视觉描述（或阶段1模型生成的描述）
- LoRA rank: 16-32
- 训练步数: 2000-5000

---

## 6. 模块对照表：旧方案 vs 新方案

| 旧方案模块 | 新方案模块 | 变化说明 |
|-----------|-----------|----------|
| `ProtTokenizer` | `ProtTokenizer` | **保留不变** |
| `EmbeddingFromPretrained` | `EmbeddingFromPretrained` | **保留不变** (冻结) |
| `AttentionCompressionFixed` (75×768) | `ProteinEncoder` (N×1536) | **重新设计**: 维度/长度/架构全部改变 |
| `sos_token` / `eos_token` | 删除 | Qwen有自己的BOS/EOS |
| `position_embedding` (77, 768) | 删除 | Qwen有RoPE位置编码 |
| `encode_protein` 中的padding到77 | 删除 | Qwen无固定长度限制 |
| `_create_causal_attention_mask` | 删除 | Qwen内部处理 |
| `clip_text_model.encoder()` | 删除 | 不再使用CLIP encoder |
| `CLIPTextModel` 整体 | 删除 | 不再需要CLIP |
| `CLIPTokenizer` | 删除 | 不再需要CLIP tokenizer |
| — | `ProteinEncoder` | **新增**: cross-attention压缩 |
| — | `Qwen2.5-1.5B + LoRA` | **新增**: 文本解码器 |
| `train_protein_clip_alignment.py` | `train_protein_text_alignment.py` | **重写**: 对齐+生成训练 |
| `train_text_to_image_lora.py` | 保留或简化 | 阶段2可复用现有代码 |
| `inference2.py` | `inference_protein2text.py` + 标准SD推理 | **重写**: 两阶段推理 |

---

## 7. 文件结构规划

```
examples/text_to_image3/
├── pretrained_encoder/           # 保留
│   ├── tokenizer.py             # ProtTokenizer (不变)
│   ├── model.py                 # EmbeddingFromPretrained (不变)
│   ├── attentionCompression.py  # 保留但不再用于新方案
│   └── protein_clip_adapter.py  # 保留但不再用于新方案
│
├── protein_text_model/          # 新增
│   ├── __init__.py
│   ├── protein_encoder.py       # ProteinEncoder (新设计)
│   ├── protein_to_text.py       # ProteinToTextModel (完整模型)
│   └── data_utils.py            # 数据合并/增强工具
│
├── train_protein_text_alignment.py  # 新增: 阶段1训练脚本
├── inference_protein2text.py        # 新增: 蛋白→文本推理
├── run_protein_text_train.sh        # 新增: 训练启动脚本
│
├── dataprocess/                  # 保留
│   ├── build_protein_text_pairs.py  # 新增: 数据合并脚本
│   ├── image_to_description.py      # 保留
│   └── ...
│
├── IMPROVEMENT_DESIGN.md         # 本文档
└── ...
```

---

## 8. 实施路线图

### Phase 0: 数据准备 (1-2天)

- [ ] 编写 `build_protein_text_pairs.py`，合并 .out 文件与 CSV 文件
- [ ] 生成 `protein_text_pairs.jsonl` 训练数据
- [ ] 统计数据分布，确认物种/描述覆盖度
- [ ] 扩展 Qwen3.6 描述生成到更多物种（如当前只有16种）

### Phase 1: 模型实现 (2-3天)

- [ ] 实现 `ProteinEncoder`
- [ ] 实现 `ProteinToTextModel`
- [ ] 编写 `train_protein_text_alignment.py`
- [ ] 编写 `inference_protein2text.py`

### Phase 2: 对齐训练 (1-2天)

- [ ] 阶段1: 对比对齐训练
- [ ] 验证蛋白embedding与文本embedding的相似度矩阵
- [ ] 调整温度参数和训练超参

### Phase 3: 生成微调 (2-3天)

- [ ] 阶段2: 生成微调训练
- [ ] 评估生成文本质量 (BLEU, ROUGE, 人工检查)
- [ ] 调整LoRA参数和训练超参

### Phase 4: 端到端验证 (1-2天)

- [ ] 蛋白ID → 文本 → SD推理 → 图像
- [ ] 对比生成图像质量
- [ ] 修复发现的问题

### Phase 5: 扩展与优化 (持续)

- [ ] 扩展到更多鸟类物种
- [ ] 尝试 SDXL/FLUX 替代 SD 1.5
- [ ] 数据增强实验
- [ ] 超参数搜索

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 16种鸟数据不足导致过拟合 | 高 | 高 | 优先扩展物种数量; 使用数据增强; 增加正则化 |
| 蛋白ID→外观描述的映射关系弱 | 中 | 高 | 验证蛋白功能组与外观特征的相关性; 考虑加入物种名作为辅助输入 |
| Qwen2.5-1.5B 生成质量不够 | 低 | 中 | 升级到 Qwen2.5-7B; 调整LoRA参数 |
| ProteinEncoder 压缩信息不足 | 中 | 中 | 增加 num_output_tokens; 增加层数 |
| 阶段1和阶段2误差累积 | 中 | 中 | 阶段1对齐质量检查; 人工检查中间文本 |

---

## 10. 参考文献

| 论文 | 会议 | 关键贡献 | 与本方案的关系 |
|------|------|----------|---------------|
| Prot2Text (Abdine et al., 2024) | AAAI 2024 | RGCN+ESM→GPT-2 蛋白功能文本生成 | 架构参考 |
| Prot2Text-V2 (Fei et al., 2025) | NeurIPS 2025 | ESM-3B→LLaMA-3.1-8B + H-SCALE对齐 | **主要参考**: 两阶段训练策略 |
| ProtT3 (2024) | ACL 2024 | ESM-2→Q-Former→LM 跨模态蛋白文本 | Q-Former设计参考 |
| Prot2Chat (Wang et al., 2025) | Bioinformatics 2025 | ProteinMPNN+LLM 早期融合 | 虚拟token压缩参考 |
| BLIP-2 (Li et al., 2023) | ICML 2023 | Q-Former跨模态对齐 | ProteinEncoder设计参考 |
| ProCyon (2024) | Harvard | 11B多域蛋白表型生成 | 大规模蛋白→文本参考 |
| InstructProtein (Wang et al., 2024) | ACL 2024 | LLM指令微调蛋白双向生成 | 指令微调格式参考 |
