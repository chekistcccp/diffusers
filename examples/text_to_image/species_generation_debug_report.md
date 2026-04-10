# 物种图像生成失败问题排查报告

## 结论摘要

当前代码里不是单一的训练超参数问题，而是存在多处会直接破坏条件信号的实现错误。最关键的两个根因如下：

1. `pretrained_encoder` 中 protein token 的 `id -> embedding row` 映射被破坏了。
2. 训练、验证、最终推理三条链路对文本编码器的使用并不一致。

这两点足以解释你观察到的两个症状：

- 生成不出有意义的物种图像：因为 U-Net 接收到的条件向量已经偏离真实物种语义。
- 物种名和图像错配：因为不同 protein token 被映射到了错误 embedding，上游语义被整体错位。

---

## 一级问题（最高优先级）

### 1. `EmbeddingFromPretrained.collect_embeddings()` 为每个物种文件都插入了一个额外零向量，导致从第二个物种开始 embedding 行号整体错位

代码位置：

- `examples/text_to_image/pretrained_encoder/model.py:62`
- `examples/text_to_image/pretrained_encoder/model.py:88`

关键逻辑：

```python
embedding_matrix = [np.zeros(shape=(vector_size, ))]
...
self.word2index[prot] = self.prot_index
...
embedding_matrix.append(embeddings)
...
self.species_embedding.extend(embedding_matrix)
```

问题解释：

- `self.species_embedding` 在初始化时已经有一个零向量：
  - `model.py:56`
- 但 `collect_embeddings()` 每处理一个 `.npy` 物种文件，又新加一个零向量到 `embedding_matrix` 开头。
- `self.prot_index` 只对真实 protein 增长，没有为这些额外零向量预留索引。

结果：

- 第一个物种恰好还能对齐。
- 从第二个物种开始，每个物种块前面都会多一个没有索引的零向量。
- 因此后续物种的 protein id 会指向错误 embedding 行。

这不是轻微噪声，而是系统性错位。

一个最小化示意：

- 初始化时已有 `row 0`
- 第一个物种块又插入一个零向量，`protein A` 还能落到正确位置
- 第二个物种块再插入一个零向量后，`protein C` 的 id 会落在这个额外零向量上，而不是它自己的 embedding

直接影响：

- 文本编码器输入的 protein 序列实际读取成了“错误蛋白 embedding 序列”。
- 物种条件与图像监督的对应关系被污染。
- 训练过程中 U-Net 学到的是错误条件分布，因此图像语义会崩。

---

### 2. `ProtTokenizer` 重新编号 vocabulary，和 embedding 文件中的原始 id 不一致，所有真实 protein token 几乎都被整体偏移

代码位置：

- `examples/text_to_image/pretrained_encoder/tokenizer.py:8`
- `examples/text_to_image/pretrained_encoder/tokenizer.py:15`
- `examples/text_to_image/pretrained_encoder/tokenizer.py:17`
- `examples/text_to_image/pretrained_encoder/inference.py:37`
- `examples/text_to_image/pretrained_encoder/inference.py:49`

关键逻辑：

```python
self.vocab_to_index = {token: (idx+1) for idx, token in enumerate(vocab)}
self.vocab_to_index[self.grp_token] = 0
self.vocab_to_index[self.pad_token] = 1
```

而 `vocab` 的来源是：

```python
with open(voc_dir , 'r') as f:
    for line in f:
        aline = line.strip().split(',')
        voc.append(aline[1])
```

`example_bird_protein_vocabulary.txt` 本身已经存了明确 id，例如：

```text
1,[PAD]
0,[GRP]
2,tr|A0A493STA2|A0A493STA2_ANAPP
3,tr|A0A493SUB9|A0A493SUB9_ANAPP
```

但 tokenizer 没有读取这些 id，而是把 token 按读入顺序重新编号。实际结果可以直接验证：

- `[PAD]`: 文件 id = 1，tokenizer id = 1
- `[GRP]`: 文件 id = 0，tokenizer id = 0
- 第一个真实 protein: 文件 id = 2，tokenizer id = 3
- 第二个真实 protein: 文件 id = 3，tokenizer id = 4

也就是说：

- 特殊 token 看起来对了
- 真实 protein 从第一个开始就整体偏移了 1

直接影响：

- 即使上面第 1 个问题不存在，这里也已经会让 protein token 对到错误 embedding。
- 当前训练实际喂给编码器的是“错位后的蛋白序列”，因此物种语义不可能稳定。

这一点与第 1 个问题叠加后，错误会进一步放大。

---

## 二级问题（会显著干扰结果）

### 3. 训练时使用自定义 `speciesModel`，但训练中的验证和最终验证仍然使用默认 Stable Diffusion 文本编码器

代码位置：

- 训练时替换文本编码器：`examples/text_to_image/train_text_to_image_lora.py:517-522`
- 中期验证创建 pipeline：`examples/text_to_image/train_text_to_image_lora.py:945-954`
- 最终验证创建 pipeline：`examples/text_to_image/train_text_to_image_lora.py:974-986`

训练时：

```python
text_encoder = speciesModel(...)
tokenizer = text_encoder.tokenizer
```

但验证时：

```python
pipeline = DiffusionPipeline.from_pretrained(...)
images = log_validation(pipeline, args, accelerator, epoch)
```

这里没有把 `pipeline.text_encoder` 替换成 `speciesModel`，也没有替换 `pipeline.tokenizer`。

结果：

- 训练时 U-Net 学的是 “protein encoder 条件空间”
- 验证时却拿默认 CLIP 文本编码器去生成

所以训练过程里看到的验证图，不能代表你真正训练出来的系统效果。

这会带来两个后果：

- 你可能误判模型没有学习，或者误判某个 checkpoint 更好。
- 即使 LoRA 学到了一点东西，验证图也可能完全不对应。

---

### 4. `speciesModel` 忽略 `attention_mask`，而压缩模块对 6000 长度序列做全量 softmax，padding 会直接参与注意力

代码位置：

- `examples/text_to_image/pretrained_encoder/inference.py:85`
- `examples/text_to_image/pretrained_encoder/tokenizer.py:152-157`
- `examples/text_to_image/pretrained_encoder/attentionCompression.py:33-37`

现状：

- tokenizer 生成了 `attention_mask`
- 但 `speciesModel.forward()` 完全没有使用它
- `AttentionCompression.forward()` 对所有 token 做 softmax，没有屏蔽 pad

关键逻辑：

```python
attn_weights = F.softmax(attn_scores, dim=-1)
compressed_output = torch.bmm(attn_weights, values)
```

影响：

- 当 `max_tokens = 6000` 时，大量 padding token 会参与注意力归一化。
- 就算 pad embedding 是零，softmax 的分母依然被 pad 位置占据，真实 token 权重会被稀释。
- 条件向量会更模糊，物种区分能力继续下降。

这通常不会像前两个问题那样“直接完全错位”，但会明显伤害条件质量。

---

### 5. `speciesModel` 和其内部模块硬编码使用 `cuda` 与固定 checkpoint 路径，训练环境和推理环境很容易不一致

代码位置：

- `examples/text_to_image/pretrained_encoder/inference.py:47`
- `examples/text_to_image/pretrained_encoder/inference.py:64`
- `examples/text_to_image/pretrained_encoder/inference.py:82`
- `examples/text_to_image/pretrained_encoder/model.py:249`
- `examples/text_to_image/pretrained_encoder/model.py:282`

问题点：

- 代码中多处 `.to('cuda')`
- `speciesModel` 初始化时固定加载：

```python
self.load_model('/home/jun/work/species_genAI/finetune/pretrain_species_encoder_susbsystems/logs/ckpts/last.ckpt')
```

影响：

- 单卡环境之外很容易出现设备不一致。
- `accelerate` 管理下的多卡训练会和这个硬编码冲突。
- 不同机器、不同 checkpoint、不同目录下，代码表现不可复现。

这类问题不一定直接导致“语义错配”，但会让实验本身不稳定且难以排查。

---

### 6. 训练时 `input_ids` 在数据预处理阶段就被强制搬到 CUDA，不符合常规 dataloader / accelerate 设备流

代码位置：

- `examples/text_to_image/train_text_to_image_lora.py:678-679`

当前实现：

```python
return torch.tensor(np.array(inputs['input_ids']), dtype=torch.int64 ).to('cuda')
```

影响：

- 如果 `num_workers > 0`，worker 进程里直接触发 CUDA 往往不稳定。
- 与 `accelerator.prepare()` 的设备管理不一致。
- 即便能跑，也会引入额外拷贝和调试困难。

这更偏工程问题，但建议一起修。

---

## 三级问题（附带缺陷）

### 7. `pretrained_encoder/dataset.py` 的 dataloader 构造函数本身有 bug

代码位置：

- `examples/text_to_image/pretrained_encoder/dataset.py:60-67`

问题代码：

```python
train_ds, valid_ds, test_ds = speciesData(datapath, train=True)
```

`speciesData(...)` 返回的是一个 `Dataset` 对象，不是三个对象。这里的写法实际上不成立。

虽然这段代码目前看起来没有被你当前 diffusion 训练脚本直接调用，但说明 `pretrained_encoder` 这部分代码整体上还存在未经严格回归验证的问题。

---

### 8. `inference2.py` 中物种名提取依赖 `'/'` 分隔符，路径如果是 Windows 反斜杠会取错

代码位置：

- `examples/text_to_image/inference2.py:33-37`

当前实现：

```python
parts = file_path.split('/')
```

在 Windows 风格路径下这会失败，虽然这不会直接改变生成内容，但会导致测试统计和结果命名错误，增加“图像和物种不匹配”的表面混乱。

---

## 为什么这些问题足以解释当前症状

### 症状 1：生成不出有意义的物种图像

根因链条：

1. tokenizer 产生的 token id 与 embedding row 不一致。
2. embedding loader 又进一步在多物种情况下插入额外错位。
3. attention compression 还会把大量 padding 混入条件表示。

最终 U-Net 看到的并不是“某个物种的蛋白条件”，而是一个被严重污染的条件向量。对于 diffusion 来说，这种条件噪声会直接表现为：

- 学不到稳定语义
- 只能生成无意义或泛化失败的鸟图
- 不同物种之间难以分开

### 症状 2：物种名与图像错配

根因链条：

1. 某个 protein token 实际取到了别的 protein embedding。
2. 一个物种的整条 proteome 序列因此对应到另一个物种或混合物种的表示。
3. LoRA 会把错误条件和图像监督绑定起来。

所以在推理时，即使输入的是 A 物种的 prompt，生成也可能偏向 B 物种或混合特征。

---

## 修复优先级建议

### P0：必须先修

1. 修正 `EmbeddingFromPretrained.collect_embeddings()`，确保 embedding matrix 中每一行都和 `word2index` 严格一一对应。
2. 修正 `ProtTokenizer`，不要重新编号，必须直接读取 vocabulary 文件中已有的 id。
3. 重新训练 encoder 和 diffusion LoRA。

如果前两个问题不修，后面继续调学习率、训练步数、rank、guidance scale 基本没有意义。

### P1：紧接着修

1. 让训练验证 pipeline 和 `inference2.py` 使用完全相同的 `text_encoder + tokenizer`。
2. 给 `AttentionCompression` 引入 `attention_mask`，屏蔽 pad token。
3. 去掉所有硬编码 `cuda` 和绝对 checkpoint 路径，统一由外部参数控制。

### P2：工程清理

1. `input_ids` 保持在 CPU，由 dataloader / accelerate 统一搬运。
2. 修复 `pretrained_encoder/dataset.py` 的拆包逻辑。
3. 用 `os.path` 或 `pathlib` 处理 `inference2.py` 的路径解析。

---

## 最终判断

当前结果差的主要原因不是“LoRA 不适合这个任务”，也不是“训练轮数不够”，而是 **条件编码链路在进入 U-Net 之前就已经被代码错误破坏了**。

最核心的问题在 `pretrained_encoder`：

- embedding 构造错位
- tokenizer 编号错位

这两个问题任何一个单独存在，都足以显著破坏训练；现在两者是叠加存在的。因此“总是生成不了有意义的物种图像，同时物种名和图像错配”是符合当前代码行为的。

---

## 建议的下一步

建议按这个顺序处理：

1. 先修 `pretrained_encoder/model.py` 的 embedding 索引构造。
2. 再修 `pretrained_encoder/tokenizer.py`，改成读取文件里的原始 id。
3. 再统一训练、验证、推理三条路径的 text encoder 使用方式。
4. 修完后从头重新训练，不要沿用旧 checkpoint。

如果需要，我下一步可以直接帮你改这几处代码，并补一个最小一致性检查脚本，自动验证：

- vocabulary 文件 id 与 tokenizer id 是否一致
- tokenizer id 与 embedding row 是否一致
- 训练和推理是否使用同一 text encoder
