# Protein Only 实验设计报告

日期：2026-05-08

## 文档目的

这份报告回答的问题是：

**如果完全不使用 CLIP 文本分支，只用蛋白信息作为唯一条件，应该怎样设计一条严谨、可解释、可验证的 Protein Only 实验路线？**

这里的重点不是“把当前代码再训练几次”，而是把 Protein Only 路线重新定义成一个清晰的研究实验。

Protein Only 路线应被理解为：

- 训练和推理时都不使用自然语言 prompt
- 不保留 CLIP text encoder 作为条件分支
- 蛋白信息是唯一外部条件来源
- 生成模型要学会根据蛋白条件输出与目标物种或目标性状一致的图像

这条路线是可研究的，但它比 `CLIP only` 或 `CLIP + protein` 更难，也更需要严格的实验设计。

## 核心结论

Protein Only 不是不能做，但必须改变实验目标和方法定义。

当前最容易犯的错误是把任务想成：

- 输入一大串蛋白 token
- 直接让扩散模型生成正确物种图像

这在工程上可以跑，但在研究上定义得太粗，导致一旦失败，很难判断到底是：

- 蛋白信息本身不够
- 条件编码方式不对
- 数据配对方式有噪声
- 扩散模型无法使用这类条件

因此，Protein Only 的实验设计应遵循以下原则：

1. 任务目标先从物种层级做起，不要一开始就要求蛋白决定所有视觉细节。
2. 输入要结构化，不要把随机大蛋白串直接当作唯一主条件。
3. 蛋白 encoder 要被设计成“生成条件编码器”，而不是“分类器顺手输出 hidden states”。
4. 训练损失不能只有 diffusion loss，必须加入条件一致性监督。
5. 评估重点应放在“条件是否真的被用到”，而不只是“图像能不能生成出来”。

## Protein Only 路线的研究问题应如何定义

我建议把 Protein Only 拆成三个层级问题，而不是一次性解决全部问题。

## 层级 1：蛋白是否足以决定物种层级生成

问题定义：

给定一个物种对应的蛋白条件，模型是否能稳定生成属于该物种的鸟类图像。

这是最基础也最重要的问题。

如果这一层都做不到，就不要过早讨论更细粒度的性状控制。

## 层级 2：蛋白分组是否能决定与可见形态相关的差异

问题定义：

给定结构化蛋白分组条件，例如羽色、喙形、体型、飞行相关 subsystem，模型是否能在图像中表现出与这些分组一致的形态变化。

这一步才真正开始接近“蛋白影响形态”的研究叙事。

## 层级 3：蛋白是否可作为完全替代文本条件的生成驱动

问题定义：

在不保留 CLIP 文本支路的前提下，蛋白 encoder 输出的条件 token 是否足以稳定驱动扩散模型进行高质量条件生成。

这一层是 Protein Only 路线的最终命题，但不应该作为第一阶段的唯一成败标准。

## Protein Only 不应该承担什么任务

为了让实验目标可学，必须明确它不承担什么。

在你的数据定义下，Protein Only 不应被要求稳定决定：

- 姿态
- 背景
- 拍摄角度
- 光照
- 个体年龄
- 雌雄差异
- 羽毛状态
- 拍摄风格

这些因素在当前蛋白条件里没有明确监督来源，或至少不是强监督来源。

因此 Protein Only 的合理目标应是：

- 物种层级一致性
- 形态相关 trait 的统计一致性
- 条件变化带来的可重复、非随机的图像变化

而不是“一条蛋白输入对应一张唯一正确照片”。

## Protein Only 的实验总路线

我建议把 Protein Only 设计成三阶段实验，而不是单阶段 end-to-end 训练。

## 阶段 A：蛋白条件表征预训练

目标：

- 从蛋白输入中学习稳定的 species / trait-aware 表征

输出：

- 一个适合生成任务的 protein condition encoder

## 阶段 B：蛋白条件到扩散条件空间的对齐

目标：

- 让 protein encoder 输出的条件 token 变得“UNet 可用”

输出：

- 一个可以直接喂给扩散模型 cross-attention 的 protein conditioning interface

## 阶段 C：Protein Only 扩散训练

目标：

- 在没有 CLIP text branch 的条件下，仅靠蛋白条件驱动图像生成

输出：

- Protein Only 生成模型

这种分阶段设计的意义是：把“懂蛋白”“懂图像”“懂扩散条件”三个问题解耦。

## 推荐的任务定义

我建议 Protein Only 的主实验先定义成：

**输入结构化蛋白条件，生成与目标物种一致、并在可见形态上与蛋白分组信号一致的鸟类图像。**

这里有三个关键词：

- 结构化
- 物种一致
- 性状一致

如果你把目标只写成“输入蛋白，生成鸟图”，实验会非常难评价。

## 输入协议应该如何设计

Protein Only 路线里，输入设计是最关键的部分之一。

当前最大的问题是：你现有 prompt 主要是大规模随机蛋白 token 串，这种设计不利于扩散模型理解条件结构。

## 当前输入的局限

当前 prompt 具有以下特点：

- 由大量蛋白 accession token 组成
- 通过 `[GRP]` 分隔 group
- 同一物种会构造很多随机子集

这在“做分类”时尚可，但对“做生成条件”不够友好，因为：

1. 可见形态相关信号容易被大量背景蛋白淹没。
2. token 的角色没有显式表达。
3. 同一物种的很多 prompt 对应相似图像，模型会学会忽略细节。

## 推荐的 Protein Only 输入结构

建议把每条 protein 条件拆成三层：

1. `species-global summary`
2. `trait / subsystem groups`
3. `background proteome summary`

这三层都可以是蛋白驱动，不需要自然语言。

### 1. species-global summary

这是从全物种蛋白集合中压缩出的少量全局 token。

作用：

- 提供物种级先验
- 帮助模型先确定“这大致是哪类鸟”

建议：

- 4 到 8 个全局 summary tokens

### 2. trait / subsystem groups

这是最有研究价值的部分。

基于你现有的 `selected_systems.json` 和 `genes_by_function` 资源，把以下 group 单独保留：

- `Plumage_coloration`
- `Bill_morphology`
- `Body_Size_Variation`
- `Diet`
- `Wing_growth_and_flightlessness`
- 其他你认为与可见性状高度相关的 subsystem

每个 subsystem 单独编码成 1 到 4 个 group tokens。

作用：

- 让模型知道哪些蛋白组对应哪些形态通道
- 让后续 ablation 和可解释性分析成为可能

### 3. background proteome summary

剩余与可见形态弱相关、但可能提供整体生物背景的蛋白集合，压成少量补充 token。

建议：

- 4 到 8 个 token

作用：

- 保留物种背景信息
- 但不让它吞掉 trait 组的信号

## 推荐的 token 角色体系

Protein Only 虽然不使用自然语言，但应使用**结构化控制 token**。

这些 token 不是普通文本，而是模型内部符号。

建议至少引入：

- `[NULL]`
- `[GLOBAL]`
- `[TRAIT_PLUMAGE]`
- `[TRAIT_BILL]`
- `[TRAIT_BODY]`
- `[TRAIT_FLIGHT]`
- `[BACKGROUND]`
- `[END]`

作用：

- 明确每组 token 的功能身份
- 让 encoder 输出具有稳定结构
- 便于 null conditioning 和 classifier-free guidance

这类 token 不违背 Protein Only 原则，因为它们不是自然语言条件，而是蛋白条件协议��一部分。

## 推荐的 encoder 设计

## Protein Only 的 encoder 不应是当前 species classifier 的直接延伸

当前 `speciesModel` 更像“用于分类的蛋白表征网络”。对于 Protein Only 生成任务，它的问题是：

- 目标偏向分类而不是生成
- 输出 token 角色不明确
- 结构更多是为了压缩，而不是为了扩散条件建模

因此，Protein Only 实验应把 encoder 重新定义为：

**protein generative condition encoder**

它的职责不是只判断 species，而是生成一串对扩散模型有意义的条件 token。

## 推荐的 encoder 组成

建议由四部分组成：

1. protein embedding layer
2. group-aware aggregation module
3. token projector
4. condition formatter

### 1. protein embedding layer

复用现有蛋白 embedding 资源是合理的。

作用：

- 把离散蛋白 token 映射到连续空间

### 2. group-aware aggregation module

这一层负责按 subsystem / global / background 结构分别编码。

可选方式：

- attention pooling
- set transformer
- group-wise attention compression

不建议继续把全序列一次性无差别压缩。

### 3. token projector

把不同 group 的表征统一映射到扩散模型所需的 hidden size。

例如：

- `hidden_size = 768` for SD1.x

### 4. condition formatter

把所有 group token 按固定顺序排成一个条件 token 序列，例如：

```text
[GLOBAL tokens][TRAIT tokens][BACKGROUND tokens][END]
```

这是 Protein Only 路线里非常关键但容易被忽略的一层。

## 输出 token 数量建议

我建议 Protein Only 首轮实验控制在固定 token 数，例如：

- 32 tokens
- 48 tokens
- 64 tokens

不要一开始就输出几千个 token。

更进一步，如果目标是替代 SD1.x 的 CLIP 条件空间，可以让总 token 数靠近其常见上限，例如不超过 77。

这样做的好处是：

- 更接近原模型习惯的条件长度分布
- 计算更稳定
- 更容易分析每组 token 的作用

## Protein Only 的扩散训练结构

Protein Only 不使用 CLIP text branch，但并不意味着只训练 encoder。

我建议的主实验结构是：

- VAE 冻结
- UNet 主体冻结
- UNet cross-attention 相关 LoRA 可训练
- protein encoder 后半部分可训练
- token projector / formatter 全部可训练

原因是：

- 只训练 encoder，UNet 很难适配全新的条件分布
- 只训练 UNet LoRA，encoder 又学不会稳定条件语言

Protein Only 的成功通常需要这两端一起适配。

## 推荐的训练阶段设计

## 阶段 A：表征预训练

目标：

- 先让 protein encoder 具备稳定的 species / trait 表征能力

建议任务：

1. species classification
2. protein-protein contrastive learning
3. subsystem-aware auxiliary objectives

说明：

species classification 只能作为起点，不能作为终点。

## 阶段 B：蛋白到图像语义对齐

目标：

- 让 protein encoder 输出更贴近图像可用语义

建议任务：

1. protein-image contrastive alignment
2. species consistency with image encoder
3. optional teacher-guided latent alignment

这里的关键思想是：

即使推理时不使用 CLIP，你也可以在训练阶段借助一个图像 encoder 或教师空间来帮助蛋白条件更快学会“如何被图像模型使用”。

这不违背 Protein Only 推理设定。

## 阶段 C：Protein Only diffusion fine-tuning

目标：

- 用 protein tokens 作为唯一条件进行图像生成

训练内容：

- diffusion denoising loss
- 条件一致性 loss
- null-conditioning / guidance 训练

## Protein Only 的损失设计

只用 diffusion loss 不够。

推荐使用以下损失组合。

## 1. 主损失：Diffusion Denoising Loss

这是必须项，不展开���

## 2. Species Consistency Loss

做法：

- 用一个冻结的鸟类图像分类器或专门的 species classifier
- 约束生成图像的物种预测与目标蛋白对应物种一致

作用：

- 最直接回答“蛋白有没有驱动物种生成”

## 3. Protein-Image Contrastive / Matching Loss

做法：

- 让匹配的 protein condition 与生成图像或真实图像在共享空间内更接近
- 不匹配 pair 更远

作用：

- 强化蛋白到图像的条件对齐

## 4. Trait Consistency Loss

做法：

- 对 plumage、bill、body-size 等 trait 设计辅助分类或属性评分器
- 约束生成图像与目标 trait group 一致

作用：

- 让“蛋白影响形态”成为可测量命题

## 5. Null-Condition / CFG Training Loss

Protein Only 也必须支持 classifier-free guidance 风格的条件控制。

因此训练时要随机将一部分样本替换成：

- `[NULL]`
- 或全部 group dropout

作用：

- 支持推理阶段 guidance
- 测试条件存在与否对生成结果的影响

## 推荐的损失优先级

第一版主实验建议用：

1. diffusion loss
2. species consistency loss
3. null-conditioning training

第二版再加：

4. protein-image alignment loss

第三版再加：

5. trait consistency loss

这样更容易逐步定位收益来源。

## 数据设计应该怎么改

## 当前数据的问题

你目前的数据生成方式会把：

- 同一物种有限图像
- 配给大量随机蛋白子集

这会严重鼓励模型学会：

- 忽略蛋白细节
- 只学物种大类模板

对于 Protein Only，这个问题比 CLIP 路线更严重，因为它唯一的条件就是蛋白。

## 推荐的数据协议

### 1. 限制每张图像绑定的蛋白条件数量

不要再让一张图对应太多随机蛋白子集。

建议：

- 每张图最多绑定 1 到 4 个条件变体

作用：

- 降低 many-to-one 噪声
- 增强条件可识别性

### 2. 把验证集做成“图像不重叠 + 条件不重叠”

验证集应同时满足：

- 验证图像不出现在训练集
- 验证 protein subsets 不与训练 subsets 完全重复

否则会高估模型能力。

### 3. 为每个物种构造固定评估条件集

建议每个物种保留固定的：

- global summary prompt
- subsystem-focused prompt
- background-only prompt
- null prompt

这样可以做稳定对比。

### 4. 将 trait-related groups 从大背景蛋白里显式分离

这点对研究很重要。

如果 plumage、bill、body-size 等相关蛋白仍然淹没在 4000 随机 token 里，模型很难显式利用它们。

## 推荐的实验组

Protein Only 不应只有一个实验，而应有一组层层推进的实验。

## PO-0：Null / Weak Control

定义：

- 不使用有效蛋白条件，只输入 `[NULL]` 或弱随机 summary

作用：

- 测试无条件时模型会生成什么
- 作为最低基线

## PO-1：当前路线复现组

定义：

- 使用现有 flat protein prompt
- 使用现有 encoder 思路

作用：

- 作为当前方法的可比较基线

预期：

- 很可能能生成鸟
- 但条件敏感性弱

## PO-2：Pooled Protein Baseline

定义：

- 所有蛋白信息压成少量全局 token
- 不区分 subsystem

作用：

- 测试“蛋白整体背景是否足以驱动物种层级生成”

## PO-3：Structured Protein Tokens

定义：

- global + subsystem + background 三层条件

作用：

- 验证结构化输入是否优于 pooled baseline

这是我建议的 Protein Only 主实验。

## PO-4：PO-3 + Alignment Pretraining

定义：

- 在正式 diffusion fine-tuning 前，先做 protein-image 对齐训练

作用：

- 测试显式对齐是否提升条件利用率

## PO-5：PO-4 + Trait Consistency Supervision

定义：

- 在生成训练中加入 trait-level 辅助监督

作用：

- 测试能否把蛋白影响推进到更细粒度形态层面

## 评估协议应该如何设计

Protein Only 路线若只靠“看图像像不像鸟”来评价，是不够的。

必须回答三类问题：

1. 条件是否被用到了
2. 是否学到了物种层级差异
3. 是否学到了 trait-level 差异

## 评估 1：物种正确率

使用冻结 species classifier 或人工评审，统计：

- top-1 species accuracy
- top-k species accuracy

这是 Protein Only 是否成立的第一指标。

## 评估 2：条件敏感性

对同一随机种子，替换不同 protein condition，观察输出是否系统变化。

如果更换条件后结果几乎不变，说明模型没有真正使用蛋白。

建议做：

- 同种不同子集
- 跨物种条件替换
- subsystem swap

## 评估 3：Null Guidance 对比

比较：

- null condition
- full protein condition

在相同 seed 下的差异程度。

作用：

- 量化条件本身的控制力

## 评估 4：Trait-Swap Counterfactual

这是 Protein Only 最有研究价值的评估之一。

做法：

- 固定 global summary
- 只替换某一个 subsystem group

例如：

- 只替换 `Plumage_coloration`
- 只替换 `Bill_morphology`

观察生成结果是否在对应视觉维度上发生可解释变化。

如果有效，这会成为非常强的研究证据。

## 评估 5：多样性与记忆风险

统计：

- 同条件多 seed 的图像多样性
- 与训练图像最近邻相似度

防止模型只是复现训练图或塌缩到模板。

## 推荐的固定评估面板

建议每次 checkpoint 固定生成如下面板：

1. 每个物种 4 条固定 full protein prompts
2. 每个物种 2 条 null prompts
3. 每个物种 2 条 subsystem swap prompts
4. 每条 prompt 固定 4 个 seeds

这样可以形成长期可比的实验日志。

## 成功标准应该如何定义

Protein Only 不能只有一个“成功/失败”标准，建议按阶段定义。

## 第一阶段成功

满足：

- 明显优于 null baseline
- 生成图像可识别为目标物种的概率显著上升
- 不同蛋白条件下输出存在稳定差异

## 第二阶段成功

满足：

- structured protein input 明显优于 flat pooled input
- 条件敏感性可重复
- subsystem swap 出现方向正确的视觉变化

## 第三阶段成功

满足：

- Protein Only 在 species-level 上接近或部分达到 CLIP only 基线
- 在 trait-sensitive 设置下具备超出 CLIP only 的可解释性

注意：

Protein Only 不一定需要在所有指标上立即超过 CLIP only，它的研究价值也可以来自：

- 更直接的生物学解释路径
- trait-level 控制能力

## 为什么 Protein Only 需要单独的推理协议

当前推理脚本 `inference2.py` 只是把第二列 `prompt_seq` 直接送进 pipeline。

对于正式 Protein Only 实验，这个协议不够。

建议推理输入显式区分：

- `global_summary`
- `trait_groups`
- `background_summary`
- `null_condition`

哪怕最终这些都被格式化成一个 token 序列，也应在实验层面保留结构信息。

否则后续无法做 subsystem 级别对照和 counterfactual 分析。

## 推荐的实验顺序

## 第一步：重做输入协议

目标：

- 从 flat protein prompt 改成 structured protein condition

不做这一步，后面的 Protein Only 很难解释。

## 第二步：先做 species-level Protein Only

目标：

- 证明蛋白 alone 能驱动物种层级生成

不要一开始就追求细粒度形态控制。

## 第三步：做 alignment-enhanced 版本

目标：

- 检验对齐训练是否提升条件利用率

## 第四步：做 trait-level counterfactual

目标：

- 证明某些 subsystem 的变化会导致可解释图像变化

## 第五步：与 CLIP only / CLIP + protein 对照

目标：

- 明确 Protein Only 的优缺点

## 风险与对应对策

## 风险 1：模型只学到物种模板，不学蛋白差异

对策：

- 减少同图像对应的蛋白 prompt 数
- 增加 subsystem swap 评估
- 加条件一致性损失

## 风险 2：protein encoder 输出对 UNet 不可用

对策：

- 单独做 protein-image 对齐阶段
- 控制 token 数和角色结构
- 训练 projector 与 formatter

## 风险 3：数据的可见性状信号不足

对策：

- 优先选与羽色、喙形、体型相关的 subsystem
- 不要把所有蛋白都视为等权条件

## 风险 4：生成质量下降，但条件可解释性提高

对策：

- 把 Protein Only 定位成研究分支而非唯一主生产路线
- 与 CLIP only 和 CLIP + protein 做明确对照

## 风险 5：结果只对 seen species 有效

对策：

- 明确区分“seen species generation”和“generalization”
- 不要在第一阶段过度承诺跨物种泛化

## 最终建议

如果你要做一条严肃的 Protein Only 路线，我建议把它定义为：

**一种不依赖文本条件、以结构化蛋白信息为唯一输入的生成实验，用于检验蛋白是否足以驱动物种层级和部分 trait 层级的图像生成。**

在实现上，我最推荐的主实验不是当前 flat protein prompt 路线，而是：

1. 结构化 protein input
2. generative protein condition encoder
3. protein-image 对齐预训练
4. Protein Only diffusion fine-tuning
5. species / trait consistency 评估

如果要把优先级再压缩成最关键的五点，就是：

1. 先改输入协议，不要再把大随机蛋白串直接当唯一主条件。
2. 先做 species-level Protein Only，再逐步进入 trait-level。
3. 不要只用 diffusion loss，必须加 species consistency。
4. Protein Only 需要独立的 null-conditioning 和 counterfactual 评估。
5. 把 Protein Only 当研究实验分支，而不是当前唯一主线。

从研究价值上看，Protein Only 最强的潜力不一定是“绝对生成效果最好”，而是：

- 条件来源纯粹
- 生物学解释路径直接
- trait-level counterfactual 更有说服力

这正是它值得做的原因。
