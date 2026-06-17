# CLIP Only 基线路线实现文档

日期：2026-05-08

## 文档目标

这份文档说明如何在当前 `examples/text_to_image` 代码基础上，构建一条 **CLIP only 基线**。

这里的 “CLIP only” 含义是：

- 使用 Stable Diffusion 原生的 `tokenizer + CLIP text encoder`
- 不使用当前自定义 `speciesModel` / `ProtTokenizer`
- 不把蛋白输入送进扩散模型
- 用标准文本 prompt 驱动生成

这条路线的目的不是替代你的蛋白研究路线，而是提供一个**强而稳定的基线**，回答两个关键问题：

1. 你的图像数据和物种标签本身是否足以让扩散模型学会“按物种生成”？
2. 后续加入蛋白分支后，性能是否真的优于一个合理的文本基线？

如果这条基线都做不起来，后续所有蛋白方案都会很难解释。

## 为什么必须先做 CLIP only 基线

你当前项目已经把 CLIP text encoder 替换成了自定义 protein encoder，所以现在很难区分：

- 问题出在数据本身
- 问题出在 protein 条件接入方式
- 问题出在训练配置

CLIP only 基线能把这些问题拆开。

如果 CLIP only 基线有效，说明：

- 图像数据至少在“物种 token -> 图像”这一步是可学的
- 扩散 LoRA 训练流程本身基本正常
- 后续蛋白路线失败，更可能是条件建模方法有问题

如果 CLIP only 基线也无效，说明优先要查：

- 图像质量
- 物种标签
- 数据切分
- 类别不平衡
- prompt 设计

## 这条路线在当前仓库里的总体策略

你现在的代码里已经有一个非常好的起点：

- `examples/text_to_image/train_text_to_image_lora.py`

它原本就是 Hugging Face diffusers 的标准 LoRA 文本训练脚本，只是你后来把 `CLIPTokenizer + CLIPTextModel` 替换成了 `speciesModel`。

因此，CLIP only 路线的本质不是重写训练器，而是：

1. 恢复原生 CLIP 条件链路
2. 准备一个文本化的数据集
3. 设计合理的 species prompt
4. 用标准 LoRA 训练与评估

## 这条路线不做什么

为了保证基线清晰，CLIP only 路线里：

- 不接入蛋白信息
- 不使用 `pretrained_encoder/`
- 不使用 `ProtTokenizer`
- 不使用 `[GRP]` 分组蛋白 prompt
- 不讨论蛋白到图像的一致性损失

否则“基线”就不再是基线。

## 推荐的基线任务定义

我建议 CLIP only 基线的任务先定义为：

**输入一个物种文本 token 或物种描述 prompt，生成与该物种一致的鸟类图像。**

注意，这个任务只检验“物种条件是否可学”，不检验蛋白条件。

这是合理的，因为在你的研究路线里，蛋白条件最终至少要做到：

- 不弱于 species-level 文本条件

如果连这一步都比不过，就很难证明蛋白接入方案有效。

## 数据应该如何准备

## 当前数据的可复用部分

你现有的数据生成脚本已经能产生：

- 图像路径
- 对应物种

这一点足够支撑 CLIP only 基线。

虽然当前训练数据第二列是蛋白序列 prompt，但图像路径本身已经包含物种目录信息，可以直接拿来生成新的文本 prompt 数据集。

例如当前路径形式类似：

```text
.../Anas_platyrhynchos/.../resized_image.png
```

这足够反推出：

- species id = `Anas_platyrhynchos`

## 推荐的数据格式

建议为 CLIP only 单独准备一份数据集，不要复用蛋白 prompt 那一列。

推荐格式仍然是 Hugging Face datasets 可直接使用的两列：

- `image`
- `text`

例如：

```text
image: /path/to/image.png
text: "a wildlife photograph of Anas platyrhynchos"
```

或者：

```text
image: /path/to/image.png
text: "<spe_anas_platyrhynchos>"
```

两种形式都可以，但建议分开做两个基线版本。

## 推荐做两个 CLIP only 子基线

### 基线 A：自然语言物种名 prompt

示例：

- `a wildlife photograph of Anas platyrhynchos`
- `a bird photograph of Gallus gallus`
- `a realistic photo of Pipra filicauda`

优点：

- 最接近标准 text-to-image
- 不需要新加特殊 token
- 实现最简单

缺点：

- CLIP 未必了解所有拉丁学名
- 不同学名在预训练语料中的覆盖程度不同

### 基线 B：专用 species token prompt

示例：

- `<spe_anas_platyrhynchos>`
- `<spe_gallus_gallus>`
- `<spe_pipra_filicauda>`

配合固定模板：

- `a wildlife photograph of <spe_anas_platyrhynchos>`

优点：

- 更适合学习数据集内的离散类别
- 更干净，利于后续与 protein branch 比较

缺点：

- 需要向 tokenizer 增加新 token，并调整 text encoder embedding

## 我更推荐哪个

如果目的是先尽快得到稳定结果，我建议：

1. 先做基线 A
2. 再做基线 B

原因很简单：

- 基线 A 更快
- 基线 B 更适合后续研究比较

## Prompt 设计建议

## 不推荐只用裸物种名

例如：

- `Anas platyrhynchos`

虽然技术上可以，但不够稳定。

建议总是加一个图像先验模板，例如：

- `a wildlife photograph of {species}`
- `a realistic bird photo of {species}`
- `a nature photo of a {species}`

这样做的作用是：

- 提供稳定的图像生成上下文
- 让模型更聚焦物种差异，而不是先猜图像类型

## 推荐的 prompt 模板

第一轮实验建议使用固定单模板，避免引入额外变量。

建议模板：

```text
a wildlife photograph of {species}
```

这里 `{species}` 可以替换成：

- `Anas platyrhynchos`
- `Gallus gallus`
- `Balaeniceps rex`

## 是否要把下划线改成空格

建议改。

即：

- `Anas_platyrhynchos` -> `Anas platyrhynchos`

原因：

- 更接近自然语言和拉丁学名表达
- 更利于 CLIP 处理

## 是否要加 common name

可以作为第二轮增强实验，但不建议一开始就加。

比如：

- `a wildlife photograph of Anas platyrhynchos, mallard`

优点：

- 有些常见鸟类 common name 在 CLIP 预训练里更熟悉

缺点：

- 你需要维护一个可靠的物种名映射表
- 变量变多，不利于第一轮基线

因此第一轮建议只用拉丁名。

## 训练脚本应该怎么组织

## 核心原则

CLIP only 路线应尽量靠近官方 diffusers 逻辑，减少自定义改动。

最理想的方式不是在当前 heavily customized 的脚本上继续堆条件，而是：

1. 使用一个恢复原始 CLIP 路径的 LoRA 训练脚本
2. 只改数据输入和少量实验参数

## 推荐的脚本来源

有两种实现方式。

### 方式 1：基于当前 `train_text_to_image_lora.py` 恢复 CLIP 分支

思路：

- 去掉 `speciesModel` 替代逻辑
- 恢复 `CLIPTokenizer.from_pretrained(...)`
- 恢复 `CLIPTextModel.from_pretrained(...)`
- 其余训练逻辑基本保持不变

优点：

- 最大限度复用你当前仓库结构
- 和现有训练脚本参数体系一致

缺点：

- 你当前文件已经被改过，恢复时要小心不要遗留 protein 逻辑

### 方式 2：单独复制一份纯官方版 CLIP LoRA 训练脚本

例如新建一个专门的脚本：

- `train_text_to_image_lora_clip_only.py`

思路：

- 以官方 diffusers 的 LoRA 脚本为基底
- 只做项目最小相关修改

优点：

- 边界清晰
- 不会和 protein 路线逻辑混在一起
- 更适合做长期对照基线

缺点：

- 需要多维护一个脚本

## 我推荐哪种方式

我更推荐：

**新建单独脚本做 CLIP only 基线。**

原因：

- 基线应该稳定、独立、可重复
- 不应与 protein 实验脚本纠缠在同一个文件里
- 后续汇报和论文复现实验也更清楚

## 数据集加载方式

建议沿用你当前项目已经兼容的 Hugging Face datasets 格式。

可以继续使用：

- `dataset_name`
- 或本地 `imagefolder` + metadata

若是本地数据，建议组织成：

```text
train_data_dir/
  metadata.jsonl
  species_xxx/
    img1.png
    img2.png
```

其中 `metadata.jsonl` 每行类似：

```json
{"file_name": "species_xxx/img1.png", "text": "a wildlife photograph of Anas platyrhynchos"}
```

这样就能直接复用 diffusers 官方数据读取方式。

## 训练参数建议

## 目标

CLIP only 基线的目标不是极限 SOTA，而是：

- 稳定
- 可解释
- 可作为后续蛋白路线对照

因此建议参数保守一些。

## 初始参数建议

如果基于 SD1.5 LoRA，可以先从以下量级开始：

- `resolution = 512`
- `train_batch_size = 4` 或 `8`
- `gradient_accumulation_steps = 1` 或 `2`
- `learning_rate = 1e-4`
- `rank = 4` 或 `8`
- `num_train_epochs = 10` 起步
- `validation_epochs = 1`

如果数据量较小：

- 优先减少学习率而不是盲目加 epoch
- 注意过拟合

## validation_prompt 应该怎么设置

建议固定若干物种 prompt，每个 epoch 或每若干步都生成同一组样例。

例如：

- `a wildlife photograph of Anas platyrhynchos`
- `a wildlife photograph of Balaeniceps rex`
- `a wildlife photograph of Gallus gallus`

这样你能直接观察：

- 模型是否区分不同物种
- 不同 checkpoint 是否在变好

## 推理脚本应该怎么设计

CLIP only 路线应有独立推理脚本，不要复用 `inference2.py`。

原因：

- `inference2.py` 当前绑定的是 protein encoder 和 `prompt_seq`
- CLIP only 推理输入应该是标准文本 prompt

推荐单独准备：

- `inference_clip_only.py`

输入形式建议支持：

1. 单条 prompt 推理
2. 一个物种列表批量推理
3. 固定 seed 的可重复生成

例如：

```text
--prompt "a wildlife photograph of Anas platyrhynchos"
```

或者：

```text
--species_file species_list.txt
```

## 评估应该怎么做

CLIP only 基线不是只看“图像能不能出”，而是要回答：

**物种条件是否真的被学到了。**

## 必做评估 1：人工可视检查

固定每个物种生成若干张图，检查：

- 是否像鸟
- 是否像该物种
- 不同物种之间是否明显可区分

这是最低限度。

## 必做评估 2：物种分类正确率

建议使用一个独立的鸟类分类器或人工标注，评价生成图像是否被识别为目标物种。

指标示例：

- top-1 species accuracy
- top-k species accuracy

这一步极其重要，因为你后续蛋白路线也需要同一套评估协议。

## 必做评估 3：条件敏感性

选几个视觉差异较大的物种，观察替换 prompt 后生成结果是否同步变化。

如果模型对 prompt 不敏感，即使图片看起来不错，也不能算有效基线。

## 建议评估 4：训练集记忆风险

你当前图像数据可能存在同物种图像数量少、重复采样多的问题。

因此建议额外检查：

- 生成图像是否只是复现训练图风格
- 是否出现明显 memorization

## 与后续蛋白路线的关系

CLIP only 基线不是终点，而是后续比较的参照物。

后面所有蛋白方案都应该回答：

- 是否优于 CLIP only？
- 优于多少？
- 优在哪个层面？

建议后续始终保留这三类对照：

1. `CLIP only`
2. `Protein only`
3. `CLIP + Protein`

其中 `CLIP only` 是最基本的 sanity check。

## 建议的实验顺序

## 第一阶段：最小可运行基线

目标：

- 恢复标准 CLIP tokenizer / text encoder
- 数据只使用 `species -> text prompt`
- 跑通训练、验证、推理

输出：

- 可生成若干物种的 LoRA baseline

## 第二阶段：强化 prompt 设计

目标：

- 比较不同 prompt 模板
- 比较拉丁名 vs 拉丁名+common name
- 比较自然语言 species prompt vs 特殊 species token

输出：

- 找到最稳的 CLIP only 表达方式

## 第三阶段：建立正式评估协议

目标：

- 固定物种列表
- 固定随机种子
- 固定评价指标

输出：

- 后续蛋白实验共享同一套 benchmark

## 风险与注意事项

## 1. CLIP 可能不熟悉全部拉丁学名

这是一个真实风险。

对策：

- 先做自然语言模板基线
- 如效果一般，再尝试 species special tokens

## 2. 物种数据本身可能不平衡

如果某些物种图像很少，CLIP only 基线也会偏。

对策：

- 先统计每个物种图像数
- 保证验证时覆盖所有物种

## 3. 不要让基线和 protein 路线共享同一套 prompt 列

CLIP only 应单独构造 `text` 字段，不要直接复用蛋白 prompt 列。

## 4. 不要一开始就追求复杂 prompt engineering

基线最重要的是控制变量。

第一轮请坚持：

- 单一模板
- 稳定训练参数
- 清晰评估

## 最终建议

如果你的目标是为后续蛋白研究提供可靠参照，CLIP only 路线应这样实现：

1. 恢复标准 CLIP tokenizer 和 CLIP text encoder。
2. 单独构造以物种文本 prompt 为条件的数据集。
3. 先用固定自然语言模板做最小基线。
4. 再做 special species token 版本作为增强基线。
5. 用独立推理脚本和统一评估协议记录结果。

我建议你把这条路线当成：

- 数据 sanity check
- 训练流程 sanity check
- 蛋白方案的主对照实验

而不是把它视为与你的研究目标冲突的“旁线”。实际上，越是想证明蛋白有价值，越需要先把 CLIP only 基线做扎实。
