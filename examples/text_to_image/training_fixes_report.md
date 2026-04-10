# 训练链路修复报告

## 本次修复的目标

这次没有继续调超参数，而是先修训练链路里会直接破坏条件语义的代码问题。目标是让训练、训练内验证和后续测试至少使用同一套、且索引对齐的物种编码器。

---

## 已修复的问题

### 1. 修复 protein token id 和 embedding row 的对齐

修改文件：

- `examples/text_to_image/pretrained_encoder/model.py`
- `examples/text_to_image/pretrained_encoder/tokenizer.py`

修复内容：

- `EmbeddingFromPretrained` 现在为 `[GRP]` 和 `[PAD]` 预留了两行零向量，保证：
  - `[GRP] -> 0`
  - `[PAD] -> 1`
  - 第一个真实 protein 从 `2` 开始
- 删除了 `collect_embeddings()` 中“每个物种文件额外插入一行零向量”的逻辑，避免从第二个物种开始 embedding 整体错位。
- `example_bird_protein_vocabulary.txt` 改为按 id 排序写出，方便检查。
- `ProtTokenizer` 不再重新给 vocabulary 编号，而是直接使用词表文件中的原始 id。

实际意义：

- 训练时输入的 protein token 现在会对到正确的 embedding 行。
- 这是本次修复里最关键的一项。

---

### 2. 把 attention mask 真正接入 species encoder

修改文件：

- `examples/text_to_image/pretrained_encoder/attentionCompression.py`
- `examples/text_to_image/pretrained_encoder/model.py`
- `examples/text_to_image/pretrained_encoder/inference.py`
- `examples/text_to_image/train_text_to_image_lora.py`

修复内容：

- tokenizer 输出的 `attention_mask` 现在会从训练 dataloader 一直传到 `speciesModel`。
- `AttentionCompression` 在 softmax 前会屏蔽 pad token。
- 空 prompt 现在会得到全 0 的 `attention_mask`，不再被错误编码成 `[UNK]`。

实际意义：

- 长度 6000 的 pad 不会再参与注意力归一化稀释真实 token。
- 训练内验证和 classifier-free guidance 的 unconditional 分支也更合理。

---

### 3. 统一训练和训练内验证使用同一套 text encoder

修改文件：

- `examples/text_to_image/train_text_to_image_lora.py`

修复内容：

- 新增了 `build_validation_pipeline()`。
- 训练中间验证和最终验证都会把 pipeline 的：
  - `text_encoder`
  - `tokenizer`
  替换成当前训练使用的 `speciesModel` 和 `ProtTokenizer`。

实际意义：

- 训练时看到的 validation 图，终于和真实训练条件空间一致了。
- 之前“训练用 species encoder，验证用默认 CLIP”的错位已经去掉。

---

### 4. 修复训练数据预处理中的设备流和 mask 丢失问题

修改文件：

- `examples/text_to_image/train_text_to_image_lora.py`

修复内容：

- `tokenize_captions()` 现在返回：
  - `input_ids`
  - `attention_mask`
- 不再在 dataset transform 阶段把 `input_ids` 强制 `.to("cuda")`。
- `collate_fn()` 会同时收集 `input_ids` 和 `attention_mask`。
- `SuperNet.forward()` 现在显式接收 `attention_mask`，并在冻结的 text encoder 上用 `torch.no_grad()` 计算条件向量。

实际意义：

- dataloader 和 accelerate 的设备管理回到了正常路径。
- 训练图像和物种条件终于是完整成对输入 U-Net。

---

### 5. 去掉训练核心路径中的 checkpoint 和配置硬编码

修改文件：

- `examples/text_to_image/pretrained_encoder/inference.py`
- `examples/text_to_image/train_text_to_image_lora.py`
- `examples/text_to_image/inference2.py`

修复内容：

- `speciesModel` 不再内部硬编码 Linux checkpoint 路径。
- 训练脚本新增参数：
  - `--species_encoder_config`
  - `--species_encoder_checkpoint`
- 训练脚本会优先使用命令行传入的 species encoder checkpoint。
- 如果 config 里的路径是相对路径，会自动按 config 文件所在目录解析。
- `inference2.py` 也新增了 `--species_encoder_checkpoint`，并且会按 config 文件目录解析路径。

实际意义：

- 训练和测试不再依赖仓库里写死的 `/home/jun/...` 路径。
- 你可以明确指定你真正要用的 species encoder checkpoint。

---

## 本次未动的内容

下面这些我没有在这一步直接改：

- `pretrained_encoder/config.yml` 里的默认路径仍然是旧的 Linux 绝对路径。
- 旧的训练输出 checkpoint 不会自动迁移或兼容这套新链路。
- `pretrained_encoder/dataset.py` 的旧 bug 还在，但它不在当前 diffusion 训练主链路里。

这意味着：

- 你下一次训练不要继续沿用旧 LoRA checkpoint 做结论。
- 你需要准备一份你本地可用的 species encoder config，或者直接在命令行传入对应 checkpoint。

---

## 已完成的本地校验

我已经在本地做了以下检查：

1. 对修改过的文件执行了 `python -m py_compile`，语法通过。
2. 用最小脚本验证了 tokenizer 现在会严格复用词表里的原始 id。
3. 验证了空字符串 prompt 现在会生成全 0 `attention_mask`，不会被误编码成未知 token。
4. 静态确认了训练脚本里 `attention_mask` 已经贯通：
   - tokenize
   - preprocess
   - collate
   - SuperNet
   - speciesModel

---

## 你接下来要做什么

### 1. 准备可用的 species encoder checkpoint

你现在必须明确一个可用的 pretrained species encoder checkpoint 路径。

训练脚本已经要求你通过以下任一方式提供它：

- 在命令行传 `--species_encoder_checkpoint`
- 或者在 species encoder 的 YAML 配置文件里加入：

```yaml
checkpoint: /your/path/to/last.ckpt
```

建议优先使用命令行显式传参，避免混淆。

---

### 2. 检查 species encoder config 里的 3 个路径

你提供给训练脚本的 species encoder config 里，至少要保证下面 3 个字段能在你本机访问：

- `embeddings`
- `species_classes`
- `vocabulary`

如果你继续用仓库自带的 `examples/text_to_image/pretrained_encoder/config.yml`，那它里面的 Linux 绝对路径大概率需要你改成你自己的本地路径。

---

### 3. 从头重新训练，不要沿用旧的 LoRA 结果做判断

因为这次修的是条件编码链路本身，所以旧训练结果不再有可比性。

建议：

- 新开一个 `output_dir`
- 不要 `resume_from_checkpoint`
- 从头训练一轮

否则你会把旧错误条件空间下学到的 LoRA 权重继续带入新链路。

---

### 4. 训练时传入新的参数

你下一次启动训练时，至少要补上：

```bash
--species_encoder_config <你的species encoder配置文件>
--species_encoder_checkpoint <你的species encoder ckpt>
```

如果配置文件已经可用，最少要确保 checkpoint 参数是明确传入的。

---

### 5. `dataloader_num_workers` 可以设为 8

当前代码在 Linux 服务器上可以直接把：

```bash
--dataloader_num_workers 8
```

用于训练，不需要再改代码。

原因：

- 训练数据预处理阶段已经不再把 `input_ids` 提前搬到 CUDA。
- `attention_mask` 和 `input_ids` 都在 CPU 侧由 dataloader worker 生成和拼 batch。
- 当前 `collate_fn()` 只是堆叠 tensor，没有额外的 GPU 逻辑。

实际建议：

- 在 Linux 上，`num_workers=8` 更准确地说是 8 个 worker 进程，不是 8 个线程。
- 如果服务器 CPU 核数、内存和磁盘吞吐足够，`8` 是合理起点。
- 如果训练日志里出现 dataloader 卡顿、CPU 满载或 I/O 打满，再回退到 `4` 做对比。

---

### 6. 补齐了训练主链路的 NPU 支持

修改文件：

- `examples/text_to_image/train_text_to_image_lora.py`
- `examples/text_to_image/pretrained_encoder/model.py`
- `examples/text_to_image/pretrained_encoder/inference.py`

修复内容：

- `torch_npu` 改成可选导入，不再在非 NPU 环境下直接 import 失败。
- 如果 `accelerate` 实际选中了 NPU，但环境里没有 `torch_npu`，训练会直接抛出明确信息。
- `EmbeddingFromPretrained` 现在会跟随 `speciesModel` 传入的实际 device，而不是内部只按 CUDA/CPU 二选一。
- 训练内验证清 cache 改成按设备类型处理，不再固定调用 `torch.cuda.empty_cache()`。

实际意义：

- 你在训练脚本里新增的 NPU 入口现在不会只停留在主脚本层面。
- `speciesModel` 内部的 embedding 层也能跟着走到 NPU。

当前仍需注意：

- `pretrained_encoder/model.py` 里有几段旧的 `cuda` 写法还留在未使用函数和示例代码里，但不在当前训练主链路上。
- 如果你后面要把整个仓库都做成“完全 NPU 化”，这些残留分支还可以再清一次。

---

### 7. 示例 embedding 文件检查结果

我检查了：

- `examples/text_to_image/embed/Aptenodytes_patagonicus.npy`
- `examples/text_to_image/embed/Aptenodytes_patagonicus.idx`

检查结果：

- `.npy` shape 为 `(12331, 1024)`，dtype 为 `float32`
- `.idx` 行数为 `12331`
- 两者条目数一致
- `.idx` 的每一行格式都能被当前 loader 正确解析出 protein token
- 我抽查的几个 token 也能在当前 `example_bird_protein_vocabulary.txt` 中找到对应 id

结论：

- 这个示例 embedding 文件本身没有明显格式问题。
- 它和当前 `EmbeddingFromPretrained.collect_embeddings()` 的读取方式是兼容的。

---

### 8. 训练后测试也要传 species encoder checkpoint

`inference2.py` 现在同样要求：

```bash
--species_encoder_checkpoint <你的species encoder ckpt>
```

否则推理时无法加载与你训练一致的物种编码器。

---

## 推荐的执行顺序

1. 先确认 species encoder checkpoint 的真实路径。
2. 再确认 config 中 `embeddings/species_classes/vocabulary` 三个路径都能访问。
3. 新开目录从头训练 diffusion LoRA，并优先尝试 `--dataloader_num_workers 8`。
4. 训练过程中观察新的 validation 图。
5. 训练结束后用更新后的 `inference2.py` 做测试。

---

## 最终判断

这次修复已经把训练链路里最关键的代码级错误处理掉了，尤其是：

- token id 和 embedding row 错位
- pad mask 未生效
- 训练和验证 text encoder 不一致
- species encoder checkpoint 硬编码

现在如果训练结果仍然不好，下一步才值得去看：

- species encoder 本身质量
- 数据集标注质量
- prompt 构造方式
- 训练超参数

在这之前，继续调学习率和训练步数的收益很低。
