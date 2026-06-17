# `examples/text_to_image` 物种条件生成问题审计报告

日期：2026-05-08

## 结论摘要

这套代码当前“训练多次仍无法稳定根据输入生成正确物种图像”，高概率不是单一超参数问题，而是以下几类问题叠加导致：

1. 条件空间不对齐：你把 Stable Diffusion 原生的 CLIP text encoder 完全替换成了一个冻结的蛋白/物种编码器，但训练时只更新了 UNet LoRA，几乎没有给系统足够能力去把“蛋白表示空间”对齐到“图像生成条件空间”。
2. prompt 接口和验证方式不一致：当前 tokenizer 只支持逗号分隔的蛋白 token，不支持自然语言 prompt 或单纯 species name。只要输入不是蛋白 token 串，就会大面积退化为 `[UNK]`。
3. 配置和数据版本明显漂移：仓库默认的 species encoder 配置仍然是 `birds_9`，但数据生成脚本已经在产出 `16birds` 数据。若你实际训练的是 16 个物种而条件编码器还是 9 物种版本，模型很难学对。
4. 数据构造方式鼓励模型忽略 prompt 细节：同一物种的有限图片被重复配给大量随机蛋白子集，最容易学到的策略是“只画一个泛化的该物种模板”甚至“忽略条件”。

基于代码静态检查，我认为最核心的根因排序是：

1. 条件编码器与 SD 条件空间未对齐。
2. 实际输入 prompt 与 tokenizer 预期不一致。
3. species set / checkpoint / vocabulary / data 版本不一致。
4. 数据配对策略信号噪声比过低。

## 检查范围

本次检查主要覆盖：

- `examples/text_to_image/train_text_to_image_lora.py`
- `examples/text_to_image/inference2.py`
- `examples/text_to_image/pretrained_encoder/*`
- `examples/text_to_image/dataprocess/*`
- `examples/text_to_image/README.md`
- `examples/text_to_image/README_sdxl.md`

说明：我没有直接复现完整训练，因为仓库中默认配置引用了仓库外部的绝对路径数据和 checkpoint，无法在当前工作区内端到端重跑；但仅从代码一致性和局部验证来看，已经存在足以解释失败现象的高风险问题。

## 主要发现

### 1. 最关键问题：你把 CLIP 条件空间整体替换掉了，但只训练了 UNet LoRA

证据：

- `examples/text_to_image/train_text_to_image_lora.py:593-601`
- `examples/text_to_image/train_text_to_image_lora.py:610-613`
- `examples/text_to_image/train_text_to_image_lora.py:623-639`
- `examples/text_to_image/pretrained_encoder/inference.py:150-176`
- `examples/text_to_image/pretrained_encoder/model.py:347-372`

代码行为是：

- 训练脚本不再加载 CLIP tokenizer / CLIPTextModel。
- 它直接把 `speciesModel` 作为 `pipeline.text_encoder`。
- 这个 `speciesModel` 在扩散训练阶段是冻结的。
- 训练时唯一真正更新的是 UNet 内部的 LoRA 层。

这意味着：

- Stable Diffusion 原本习惯接收的是 CLIP text embedding 分布。
- 现在你给它的是一个“蛋白 token -> AttentionCompression -> 75x768” 的自定义表示。
- 这套表示虽然形状上能接进 UNet cross-attention，但语义空间完全不是 CLIP 空间。
- 而你没有训练一个足够强的“对齐层”去桥接这两个空间，只让 UNet LoRA 去硬适应。

这通常会导致两种结果：

- loss 可以下降，但模型学到的是很弱的条件依赖；
- 生成结果更像“底模先验 + 少量风格偏移”，而不是“稳定按物种条件生成”。

这不是简单调大 `rank`、`lr`、`epochs` 就能根治的问题，属于方法级别失配。

### 2. 当前 tokenizer 不支持自然语言或 species name；这会直接让 prompt 失效

证据：

- `examples/text_to_image/pretrained_encoder/tokenizer.py:67-98`
- `examples/text_to_image/pretrained_encoder/tokenizer.py:112-169`
- `examples/text_to_image/train_text_to_image_lora.py:740-768`
- `examples/text_to_image/inference2.py:94-96`

`ProtTokenizer` 的规则非常明确：

- 它不是自然语言 tokenizer。
- 它只是把字符串按逗号 `,` 切分。
- 每个切分后的片段必须是词表里的蛋白 token。
- 否则就变成 `[UNK]`。

我做了一个本地快速验证，结果如下：

- 输入 `cute dragon creature`
  - 编码后 `input_ids` 基本就是 `[UNK] + PAD...`
  - 解码结果是 `[UNK]`
- 输入一串合法蛋白 token
  - 可以正确编码和解码

这说明：

- 如果你的训练监控 `--validation_prompt` 用的是自然语言，验证图几乎没有参考意义。
- 如果你推理时输入的是 species 名称、自然语言描述、中文描述，当前代码设计上就不支持。
- 这也解释了“根据输入生成物种图像”为什么经常失败：这里的“输入”在当前实现里，必须是蛋白 token 序列，不是物种名字，也不是文本描述。

额外提醒：

- 上游 README 里仍保留了原始 diffusers 的自然语言示例，例如 `cute dragon creature`、`A naruto...`；但你已经替换了 tokenizer/text encoder，这些文档示例对当前分支是误导性的。

### 3. species encoder 默认配置仍是 `birds_9`，但数据生成脚本已经明显转向 `16birds`

证据：

- `examples/text_to_image/pretrained_encoder/config.yml:3-11`
- `examples/text_to_image/pretrained_encoder/example_species_id.json:2-10`
- `examples/text_to_image/dataprocess/prepare_dataset_birds_group_proteins_resize_512.py:34`
- `examples/text_to_image/dataprocess/pre_select_proteins.py:144-160`

仓库中的默认配置存在明显版本漂移：

- `config.yml` 指向的是 `birds9_subsystems_pretrain_...`
- `example_species_id.json` 只有 9 个物种
- 但数据构造脚本文件名和物种列表已经是 `16birds`

如果你现在的扩散训练数据来自 `large_species_img_w_prot_list_16birds_subsystems_resize_512.*`，但条件编码器仍然沿用 `birds9` 版本，那么会有几个直接后果：

- 条件编码器并不是在和扩散训练同一物种集合上训练的。
- 新增物种没有可靠的判别表示。
- 即使 forward 能跑通，hidden states 也不一定对这些物种有足够区分力。

这类“能跑但语义错位”的问题，往往比直接报错更难发现。

### 4. 数据配对方式很容易让模型学会“忽略 prompt 细节”

证据：

- `examples/text_to_image/dataprocess/prepare_dataset_birds_group_proteins_resize_512.py:34`
- `examples/text_to_image/dataprocess/prepare_dataset_birds_group_proteins_resize_512.py:62-83`
- `examples/text_to_image/dataprocess/pre_select_proteins.py:273-327`

当前数据构造逻辑是：

- 每个物种采样大量不同的蛋白子集，`sampling_size = 8000`。
- 同一物种的图片数量通常远少于 8000。
- 所以脚本会把同一批图片重复复制很多次，再随机对应到不同的蛋白子集。

这会产生一个训练偏置：

- 对同一物种而言，很多差异很大的 prompt 会对应到相似甚至相同的图像。
- 从优化角度看，最容易学到的策略不是“理解每个 prompt 的细粒度差异”，而是“把这堆 prompt 粗暴映射到一个共享物种模板”。

如果你的目标只是“同物种大类识别式生成”，这仍可能工作一点；
但如果你的目标是“输入不同物种相关条件后可靠地产生正确物种外观”，这种 many-to-one 配对方式会显著削弱 prompt 的有效性。

### 5. species encoder 训练代码本身存在明显不同步，降低了前置编码器可信度

证据：

- `examples/text_to_image/pretrained_encoder/trainer.py:47-51`
- `examples/text_to_image/pretrained_encoder/model.py:20-29`
- `examples/text_to_image/pretrained_encoder/trainer.py:73`
- `examples/text_to_image/pretrained_encoder/trainer.py:105-116`
- `examples/text_to_image/pretrained_encoder/run.py:24-31`

我看到至少以下问题：

- `trainer.py` 调用 `EmbeddingFromPretrained(..., species_id=..., voc_file=...)`，但当前 `model.py` 中的 `EmbeddingFromPretrained.__init__` 并不接受这两个参数。
- `trainer.py` 里多处硬编码 `.to('cuda:0')`，设备管理非常脆弱。
- `run.py` / `trainer.py` 使用的是当前代码之外的一套假设，说明这部分代码存在版本漂移。

这意味着：

- 即使扩散训练脚本现在能跑，前置 `species encoder checkpoint` 的训练流程也不一定和当前仓库代码严格一致。
- 你现在使用的 encoder checkpoint 很可能是由“另一版代码”训练出来的。
- 一旦 checkpoint、vocab、species_classes、当前 forward 逻辑四者不完全对应，条件表征质量会不稳定。

### 6. 默认配置与路径高度依赖外部环境，复现实验容易跑偏

证据：

- `examples/text_to_image/pretrained_encoder/config.yml:3-9`
- `examples/text_to_image/dataprocess/transfer/save-hf-parquet-512.py:30`
- `examples/text_to_image/dataprocess/prepare_dataset_birds_group_proteins_resize_512.py:30-31`

很多路径是外部绝对路径，例如：

- `/home/jun/...`
- `D:\Diffusion\...`

风险是：

- 你以为自己用的是 A 版数据 / B 版 checkpoint，实际可能读到了旧文件。
- 迁移环境后容易出现“脚本跑通，但实验对象不是你以为的那份数据”。

对于这种高度自定义的数据链路，这类配置漂移会直接影响结果可信度。

## 哪些问题最可能直接导致“生成不出正确物种”

如果只按“对最终失败现象的解释力度”排序，我建议这样看：

### 一级根因

- 条件编码器空间与 Stable Diffusion 原始 CLIP 条件空间完全不对齐，但你只训练了 UNet LoRA。
- 你在验证或推理时输入的 prompt 若不是蛋白 token 串，而是自然语言/物种名，那么 prompt 实际已经失效。

### 二级根因

- species encoder 的 species set 与当前扩散训练数据不一致，尤其是 `birds9` 对 `16birds`。
- 蛋白子集到图片的 many-to-one 配对，鼓励模型忽略 prompt 细节。

### 三级风险

- 前置 species encoder 的训练代码与推理代码不同步。
- 绝对路径和旧配置容易造成数据/权重版本错用。

## 建议的修复顺序

### 方案 A：先做一个可验证的强基线

这是我最建议的路线，因为它能最快确认问题到底在“数据”还是“自定义蛋白编码器”。

1. 先不要替换 CLIP text encoder。
2. 先用标准 CLIP tokenizer/text encoder，给每个物种分配一个稳定的离散 token，例如 `<spe_anas_platyrhynchos>`。
3. 训练一个普通的文本条件 LoRA 基线，输入只用 species token。
4. 看模型能否先学会“按物种 token 出图”。

如果这个基线都做不到，优先检查图像数据本身和物种标签；
如果这个基线能做到，说明主要问题就在你当前的蛋白编码器接入方案。

### 方案 B：如果必须使用蛋白条件，不要只训 UNet LoRA

建议至少做下面一项，而不是仅靠 UNet LoRA：

1. 在 `speciesModel` 输出后增加一个可训练投影/adapter，使其显式对齐到 CLIP 条件空间。
2. 允许 text side 的一部分参数训练，例如 adapter、projection、轻量 cross-attn bridge。
3. 如果资源允许，至少联合训练“protein projector + UNet LoRA”，而不是只训练 UNet LoRA。

当前最缺的是“对齐能力”，不是单纯“容量更大一点”。

### 方案 C：把 prompt 协议收紧并做强校验

建议立刻加三项检查：

1. 训练前随机抽 10 条 prompt，统计 `[UNK]` 比例。
2. 验证 prompt 必须来自真实蛋白 token 序列样本，不允许自然语言。
3. 推理接口明确区分：
   - `protein_prompt`
   - `species_name`
   - `natural_language_prompt`

当前这三类输入在代码里被混成了一个 `prompt`，但 tokenizer 实际只支持第一类。

### 方案 D：统一 species encoder 的数据版本

必须保证以下四项来自同一实验版本：

1. `species_classes`
2. `vocabulary`
3. `species_encoder_checkpoint`
4. 扩散训练数据中实际出现的物种集合

特别是如果你现在训练的是 16 物种，就不要继续使用仓库里这个默认 `birds9` 配置。

### 方案 E：重做验证集与评估协议

当前建议的验证方式应至少包含：

1. 固定每个物种 5 到 10 条真实 protein prompt。
2. 每次 checkpoint 都对同一组 prompt 生成图像。
3. 用同一个物种分类器或人工评审统计 top-1 物种正确率。
4. 不再使用自然语言 validation prompt。

否则你看到的只是“生成图像看起来像不像鸟”，而不是“是否按条件生成正确物种”。

### 方案 F：降低数据构造中的 prompt 噪声

建议控制以下变量：

1. 每个物种不要对同一张图重复绑定过多随机蛋白子集。
2. 先用更稳定的 species-level prompt 做实验，再逐步增加 protein subset 复杂度。
3. 对每个物种先构造少量但高一致性的 prompt，而不是一次上 8000 个随机子集。

在当前阶段，减少噪声比继续加数据量更重要。

## 可立即执行的最小排查清单

建议你按这个顺序验证：

1. 确认推理输入到底是不是合法蛋白 token 串，而不是物种名或自然语言。
2. 打印 100 条训练 prompt 的 `[UNK]` 占比。
3. 确认扩散训练数据中的物种集合和 `species_classes` 完全一致。
4. 确认 `species_encoder_checkpoint` 就是在这同一批物种上训练出来的。
5. 先做“CLIP + species token”的简单基线，验证图像数据是否可学。
6. 若基线可学，再回头做 protein encoder 对齐实验。

## 结语

从当前代码状态看，最像“主因”的不是学习率、rank、epoch 数量，而是方法链路本身：

- 输入协议和 tokenizer 不一致。
- 条件编码器与扩散模型条件空间不一致。
- species encoder 与当前数据版本可能不一致。

如果这些问题不先收敛，继续重复训练大概率只是在重复放大同一个系统性误差。
