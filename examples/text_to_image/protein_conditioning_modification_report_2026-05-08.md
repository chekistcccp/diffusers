# 保留蛋白信息前提下的改造方案报告

日期：2026-05-08

## 结论摘要

如果你的研究目标明确要求“必须使用蛋白信息”，我不建议继续沿用当前这条路线：

- 直接用蛋白编码器替换 Stable Diffusion 的 CLIP text encoder；
- 冻结蛋白编码器；
- 只训练 UNet LoRA。

这条路线的根本问题不是“蛋白信息不重要”，而是“蛋白信息被错误地接入了扩散模型”。

更合理的方向是：

1. 保留 Stable Diffusion 原生的 CLIP text branch，维持底模已有的图像先验和条件空间。
2. 将蛋白信息作为第二条件分支，而不是替代文本分支。
3. 在蛋白编码器和扩散模型之间增加一个可训练的对齐模块，把蛋白表示映射成适合扩散模型 cross-attention 使用的条件 token。
4. 训练时不仅优化扩散去噪损失，还要加入“蛋白条件是否真的影响生成结果”的一致性损失。

如果只保留一句建议，那就是：

**不要再把 protein encoder 当成 CLIP text encoder 的直接替身，而应该把它改成一个独立的 protein adapter / protein condition branch。**

## 你现在真正要解决的，不是“是否使用蛋白”，而是三个更具体的问题

### 1. 蛋白信息以什么形式进入扩散模型

当前做法是“完全替代 CLIP 文本条件”。这会把底模原本已经学好的条件空间直接打掉。

更合理的做法是让蛋白信息以以下形式之一进入：

- 独立的 protein conditioning tokens
- 独立的 protein cross-attention branch
- 独立的 protein adapter residual

### 2. 蛋白条件如何与图像语义对齐

蛋白 token 序列和 CLIP 文本 token 不在同一个语义空间。即使维度都是 `768`，它们也不是同一种“语言”。

因此必须显式学习一个桥接模块，而不是只靠 UNet LoRA 被动适配。

### 3. 你的数据里，蛋白到底能决定什么

这点非常关键。

如果当前蛋白输入是“物种级 proteome 的随机子集”，而图像是“该物种的各种个体照片”，那么蛋白条件大概率只能稳定决定：

- 物种层级特征
- 部分与可见形态相关的通路或功能群特征

它不能稳定决定：

- 姿态
- 背景
- 光照
- 个体年龄
- 雌雄差异
- 羽毛状态

也就是说，你的任务本质上更像：

**protein -> species / morphology prior -> image**

而不是：

**protein -> exact photo**

如果这个可识别性边界不先明确，后面无论怎么改模型，都会把很多不可学的随机性误当成“模型没学会”。

## 推荐的总体架构

我建议采用“双分支条件 + 可训练桥接”的结构。

### 核心思路

保留两条条件路径：

1. `CLIP text branch`
2. `protein branch`

其中：

- `CLIP text branch` 负责维持 Stable Diffusion 已有的图像语义先验
- `protein branch` 负责注入你研究真正关心的生物学条件

最终在 UNet 的 cross-attention 中融合两路条件，而不是让其中一路完全替代另一路。

### 推荐结构图

```text
protein groups / proteome
        |
        v
protein encoder
        |
        v
protein projector / adapter
        |
        +------> protein tokens / protein K,V

optional text scaffold ("a bird photo", species token, null text)
        |
        v
CLIP tokenizer + CLIP text encoder
        |
        v
CLIP text tokens

CLIP tokens + protein branch
        |
        v
UNet cross-attention
        |
        v
diffusion denoising
```

## 为什么我推荐“保留 CLIP + 加 protein branch”

### 原因 1：底模的图像先验依赖 CLIP 条件空间

SD1.x 或 SDXL 之所以能稳定响应 prompt，不只是因为 UNet 强，而是因为训练时长期依赖特定的文本条件分布。

当前代码把这个条件分布完全换掉，相当于：

- 图像先验还在；
- 条件语义坐标系被整个重写；
- 但只给了 UNet LoRA 很少的自由度去适配。

这是最不经济也最不稳定的方案。

### 原因 2：你研究需要“使用蛋白”，不等于“必须抛弃文本分支”

研究要求是蛋白必须参与条件生成，但不要求必须把文本通道完全删掉。

相反，保留 CLIP 分支有两个好处：

- 让模型继续站在一个稳定的图像语义底座上；
- 让蛋白分支专注学习“额外提供什么信息”，而不是从零承担全部条件建模。

### 原因 3：这样更容易做消融实验

如果采用双分支结构，你可以很清楚地比较：

- 只有 CLIP 分支
- 只有 protein 分支
- CLIP + protein 分支

这对研究论文也更有说服力，因为你能定量证明蛋白信息到底带来了什么增益。

## 推荐的 protein 分支设计

这里我按改造强度分三档。

## 方案 A：最小改造版

这是最适合尽快验证方向的一版。

### 做法

1. 保留当前 protein encoder，先不完全推翻。
2. 让 protein encoder 输出固定数量的 protein tokens，例如 8、16 或 32 个，而不是直接拿当前 75 个压缩 token 生硬替代 CLIP。
3. 增加一个 trainable projector，把 protein tokens 投影到 UNet 需要的 hidden size。
4. 将这些 protein tokens 和 CLIP tokens 一起送入 cross-attention。

### 融合方式

最简单的是：

- `encoder_hidden_states = concat(clip_tokens, protein_tokens)`

优点：

- 改动最少
- 容易做第一轮验证

缺点：

- 两种 token 会直接混在一起
- 如果 protein tokens 分布不稳定，可能干扰文本分支

### 适用场景

- 你想快速验证“蛋白信息是否能通过额外 token 进入扩散模型”
- 你想尽量复用当前 `speciesModel` 和 `AttentionCompression`

## 方案 B：推荐研究版

这是我最推荐的方向。

### 核心思想

不要把 protein tokens 和 CLIP tokens 简单拼接，而是像 `IP-Adapter` 那样，给 protein branch 单独一套 cross-attention 条件入口。

换句话说：

- 文本分支继续走原本的 text cross-attention
- 蛋白分支提供额外的 K/V 或 residual conditioning
- UNet 内部通过可训练门控来决定在不同层使用多少蛋白信息

### 结构建议

protein branch 包含：

1. `protein encoder`
2. `protein projector`
3. `protein attention adapter`
4. `layer-wise gates`

可以把它理解为：

- CLIP branch 管“图像是什么类型的东西”
- protein branch 管“这个鸟应该更像哪个物种 / 哪些形态特征更强”

### 为什么这个方案更稳

优点：

- 不会破坏原有 CLIP 条件通道
- 蛋白分支作用路径更清晰
- 更容易分析 protein 信息到底在哪些层起作用
- 更适合做论文里的结构解释

缺点：

- 设计和实现复杂度更高

### 我对这个方案的判断

如果你的目标不只是“先跑通”，而是希望研究结果在方法上站得住，这条路线最合理。

## 方案 C：强研究探索版

这是高风险高投入方案，不建议作为第一步。

### 思路

不直接把蛋白条件送进扩散 UNet，而是先训练一个中间语义空间：

- `protein -> morphology embedding`
- `morphology embedding -> diffusion conditioning`

或者：

- `protein -> pseudo text tokens / pseudo prompt embedding`
- `pseudo prompt embedding -> SD`

### 为什么风险高

因为这会引入更多中间模块，训练变长，调试难度更大。

### 什么时候值得考虑

只有在你已经验证：

- 双分支结构有效
- 蛋白确实能稳定提升物种一致性

之后，再考虑往更强的中间语义建模推进。

## 我最推荐的输入表示方式

我不建议把“随机 4000 个蛋白 token”原样当作条件主体。

更合理的做法是把蛋白信息结构化。

## 推荐：分层 protein prompt

你现在的数据处理中已经有一个很重要的资源：

- `genes_by_function`
- `selected_systems.json`
- `[GRP]` 分组逻辑

这其实非常适合做“层次化蛋白条件”。

### 建议把蛋白输入拆成三层

1. `species anchor`
2. `trait / subsystem groups`
3. `background proteome summary`

### 1. species anchor

这是一个显式的物种锚点，作用是告诉模型：

- 当前蛋白来自哪个物种

即使蛋白本身理论上可推断物种，我仍然建议把物种作为显式条件保留，因为：

- 它能稳定训练
- 它能把“物种层级信息”和“物种内蛋白细节信息”分开
- 便于后续做消融

### 2. trait / subsystem groups

这是最有研究价值的部分。

你当前已经按功能分组，例如：

- `Plumage_coloration`
- `Bill_morphology`
- `Body_Size_Variation`
- `Diet`
- `Wing_growth_and_flightlessness`

这些 group 不应该只是数据预处理时的分隔符，而应该成为一等条件实体。

也就是说，模型看到的不是一长串混合蛋白，而应该是：

- 一组“羽色相关 token”
- 一组“喙形相关 token”
- 一组“体型相关 token”
- 一组“飞行相关 token”

这样做的好处是：

- 条件更可解释
- 更接近“蛋白影响形态”的研究叙事
- 更容易做通路级 ablation

### 3. background proteome summary

剩余大规模随机蛋白集合可以保留，但不要让它成为唯一主体。

建议把它压成少量 summary tokens，例如：

- 4 个全局 token
- 8 个全局 token

让它提供补充背景，而不是淹没可见形态相关的显式 group 信号。

## 推荐的 protein encoder 改法

## 现有 encoder 可以保留什么

你当前的 `EmbeddingFromPretrained + AttentionCompression + DAN` 不一定要全部推倒。

可保留的部分：

- 蛋白 embedding 表
- 基础 tokenization 协议
- group-aware 输入格式

不建议原封不动保留的部分：

- 把压缩后的 token 直接当成 CLIP text hidden states
- 完全冻结整个 protein encoder
- 只保留 species classifier 预训练目标

## 更合理的 encoder 目标

protein encoder 不应该只被训练成“分类器”，而应该被训练成“可用于生成条件的表征器”。

至少要让它兼顾三类能力：

1. 区分物种
2. 表征与可见性状相关的 group 差异
3. 输出适合 diffusion conditioning 的 token 表示

## 训练时推荐的可训练范围

我建议分阶段控制可训练参数。

### 第一阶段

冻结：

- SD backbone 主体
- CLIP text encoder
- 蛋白 embedding 底层

训练：

- protein projector
- protein adapter
- UNet LoRA

目的：

- 先验证 protein 条件能否进入扩散模型而不破坏底模

### 第二阶段

在第一阶段有效的前提下，再部分解冻：

- `AttentionCompression`
- protein encoder 的最后一层或最后几层

目的：

- 让蛋白表示更贴合生成任务，而不是只贴合先前分类任务

### 第三阶段

如果前两阶段已经稳定，再考虑：

- 轻量微调更多 protein branch 参数
- 增加 layer-wise gates 或更细粒度融合

目的：

- 提升上限，而不是一开始就把系统变得不可控

## 推荐的训练目标

如果只用 diffusion denoising loss，你很难证明模型真的在使用蛋白。

因此我建议���少使用两类损失。

## 1. 主损失：扩散去噪损失

这是基础项，不多说。

## 2. 条件一致性损失

必须增加，用来约束“生成结果要与蛋白条件一致”。

可以考虑三类形式。

### A. 物种一致性损失

做法：

- 用一个冻结的鸟类图像分类器或 species classifier
- 让生成图像在 species 预测上匹配蛋白对应物种

优点：

- 最直接
- 最容易落地

缺点：

- 主要约束的是物种层级，不一定能约束细粒度形态

### B. 蛋白-图像对齐损失

做法：

- 将蛋白 embedding 与图像 embedding 做对比学习或匹配学习
- 让匹配的 protein-image 对距离更近，不匹配对更远

优点：

- 能更直接证明蛋白信息被编码进图像

缺点：

- 需要稳定的图像表征空间

### C. 通路 / 性状一致性损失

做法：

- 针对 `Plumage_coloration`、`Bill_morphology` 这类功能组，训练辅助分类器或属性预测器
- 让生成图像在这些属性上与蛋白 group 信号保持一致

优点：

- 研究解释力最强

缺点：

- 需要额外属性标签或可靠 proxy

## 我建议的损失组合

第一版优先使用：

1. diffusion loss
2. species consistency loss
3. optional protein-image contrastive loss

等系统稳定后，再考虑 trait-level loss。

## 为什么你的数据协议也必须改

如果架构改了，但数据协议不改，模型仍然可能学不到你真正想要的蛋白信号。

## 当前数据的主要问题

你现在的配对方式是：

- 同一物种采样大量随机 protein subsets
- 同一物种有限图片被重复绑定到这些 subsets

这会造成一个训练捷径：

- 模型只学物种模板，不学蛋白差异

## 数据协议的修改方向

### 1. 从“随机蛋白集合”改为“结构化 group 条件”

优先把与可见性状相关的 group 单独保留，不要让它们淹没在全量随机蛋白里。

### 2. 对 protein 条件做显式 dropout

训练时要随机丢弃：

- 全部 protein branch
- 部分 subsystem groups
- 全局 proteome summary

目的：

- 支持 classifier-free guidance 风格的条件控制
- 量化每一部分蛋白信息的贡献

### 3. 做更严格的训练/验证切分

建议至少保证：

- 验证图像不与训练图像重复
- 验证 protein subsets 不与训练 subsets 完全重复

否则容易高估条件学习效果。

### 4. 加入“近邻难负样本”

如果两个物种外观相近，训练中应该显式增加它们之间的区分压力。

对研究很重要，因为真正难的就是细粒度相似物种。

## 推理协议也要改

如果研究要求“必须使用蛋白”，推理接口必须明确。

## 建议的推理输入格式

不要再把所有输入都叫 `prompt`。

建议明确分成：

1. `species_anchor`
2. `protein_groups`
3. `global_protein_summary`
4. `text_scaffold`

其中：

- `text_scaffold` 可以是固定模板，例如“a wildlife photograph of a bird”
- 物种差异和生物学差异由 protein branch 决定

这样做的好处是：

- 推理协议清晰
- 更符合研究叙事
- 更容易测试“去掉蛋白后性能下降多少”

## 评估方案建议

如果你后面要做研究报告或论文，建议从一开始就按下面的方式设计评估。

## 必做指标

1. 物种正确率
2. 生成图像与蛋白条件的一致性
3. 同物种内多样性
4. 不同蛋白输入对生成结果的敏感性

## 建议的实验组

### E0

只有 CLIP / species token，不用蛋白 branch。

作用：

- 作为基线

### E1

CLIP + protein pooled vector

作用：

- 验证最小蛋白接入是否有效

### E2

CLIP + subsystem tokens + global protein summary

作���：

- 验证结构化蛋白条件是否优于简单 pooled vector

### E3

E2 + protein consistency loss

作用：

- 验证额外一致性监督是否真的提升条件使用程度

### E4

E3 + 部分解冻 protein encoder 后层

作用：

- 验证生成任务反向调整 protein 表征是否有收益

## 你最应该避免的几件事

### 1. 不要继续“完全替换 CLIP”

这是当前最核心的结构性问题。

### 2. 不要只训练 UNet LoRA

蛋白条件到扩散条件空间之间必须有专门的可训练桥接模块。

### 3. 不要再用自然语言验证 prompt 检查这条分支

如果 protein tokenizer 不支持自然语言，这样的验证图没有结论价值。

### 4. 不要让随机 4000 蛋白集合成为唯一主条件

这会让与可见性状真正相关的信号被噪声淹没。

### 5. 不要把“物种级 proteome”误当成“个体级图像决定因素”

否则你会对模型提出超出数据可识别性的要求。

## 我建议的研究路线

如果你要一条现实可行、同时保留研究价值的路线，我建议这样推进。

## 第一阶段：证明蛋白能被有效接入

目标：

- 不再替换 CLIP
- 让 protein branch 以独立条件进入 UNet
- 验证是否比无蛋白基线更好

建议结构：

- CLIP text branch 保留
- protein pooled / compressed tokens
- protein projector + adapter
- 训练 projector + adapter + UNet LoRA

## 第二阶段：证明结构化蛋白条件优于简单蛋白向量

目标：

- 验证 subsystem groups 是否比单个 pooled vector 更有效

建议结构：

- species anchor
- trait-related group tokens
- global protein summary tokens

## 第三阶段：证明蛋白确实影响了可见形态

目标：

- 不只是“更像某种鸟”
- 而是“与指定蛋白相关的形态特征更一致”

建议：

- trait-level 一致性评估
- 对特定 subsystem 做消融

## 最终建议

如果你的研究必须使用蛋白信息，我建议你把问题重新定义为：

**在保留 Stable Diffusion 原始图像语义底座的前提下，如何把结构化蛋白条件作为独立分支注入生成模型，并证明它提升了物种或形态一致性。**

这比“把 protein encoder 直接替代 text encoder”更稳，也更符合研究逻辑。

我对优先级的建议如下：

1. 保留 CLIP，新增 protein branch，这是第一优先级。
2. 蛋白分支使用可训练 projector / adapter，而不是只靠 UNet LoRA 适配。
3. 输入从“随机大蛋白串”改成“species anchor + subsystem groups + global summary”的层次结构。
4. 训练时加入 species consistency / protein-image alignment 监督。
5. 用明确的 ablation 证明蛋白分支确实提供了额外信息。

如果后续你要继续推进，我建议下一步不是直接改全部代码，而是先把这个报告收敛成一份更具体的“实验设计文档”：

- 先选定一种推荐结构
- 明确训练哪些模块
- 明确每个实验组的输入协议、损失和评价指标

这样后面再动代码，方向会更稳。
