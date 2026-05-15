# Qwen3-4B 金融知识微调项目

本项目用于微调 Qwen3-4B 模型，提升其金融领域知识能力。

## 项目结构

```
flun_train/
├── generate_data.py              # 数据采样脚本（统一生成训练集和测试集）
├── finetune_data_final/          # 微调数据集（由 generate_data.py 生成）
│   ├── train.json                # 训练集（2000条，金融占比80%）
│   ├── mmlu_test.json            # MMLU 测试集（500条）
│   ├── ceval_test.json           # C-Eval 测试集（500条）
│   ├── finance_test.json         # Fin-Eva 测试集（1000条）
│   └── dataset_info.json         # LLaMA-Factory 数据集配置
├── financial_evaluation_dataset/ # 金融评估数据集（Fin-Eva）
├── mmlu/                        # MMLU 通用知识数据集
├── ceval/                        # C-Eval 中文通用知识数据集
└── README.md                     # 项目文档
```

## 脚本说明

### generate_data.py

**作用**：从 Fin-Eva、MMLU、C-Eval 三个完整数据集中随机采样，生成训练集和测试集，统一转换为 OpenAI messages 格式。

**运行方式**：在项目根目录下执行 `python generate_data.py`

**依赖**：pandas、pyarrow（读取 Parquet 文件）

**运行前提**：需要先下载三个原始数据集到项目根目录（见下方"数据集下载"）

**输出**：在 `finetune_data_final/` 目录下生成以下文件：
- `train.json` — 训练集
- `mmlu_test.json` — MMLU 测试集
- `ceval_test.json` — C-Eval 测试集
- `finance_test.json` — Fin-Eva 测试集
- `dataset_info.json` — LLaMA-Factory 数据集配置

**注意事项**：
- 随机种子固定为 `seed=42`，确保结果可复现
- 金融原始数据约 930 条，脚本会自动重复 1 次增强后再采样
- Fin-Eva 的 SUFE test 分集没有 answer 列，因此不参与采样
- Ant 子集中"执业医师"和"执业药师"两个文件与金融无关，已自动排除

## 数据集

### 原始数据集

| 数据集 | 本地路径 | 文件格式 | 说明 |
|--------|----------|----------|------|
| Fin-Eva | `financial_evaluation_dataset/` | CSV | 金融评估数据集，含 Ant 和 SUFE 两个子集 |
| MMLU | `mmlu/` | Parquet | 英文多学科知识，57个学科，含 dev/test/val 三个分集 |
| C-Eval | `ceval/` | Parquet | 中文多学科知识，52个学科，含 dev/test/val 三个分集 |

### 数据集下载

MMLU 和 C-Eval 来自 ModelScope，需要先安装 git-lfs（`git lfs install`），然后分别克隆：
- C-Eval：`https://www.modelscope.cn/datasets/evalscope/ceval.git`
- MMLU：`https://www.modelscope.cn/datasets/cais/mmlu.git`
- Fin-Eva：`https://github.com/alipay/financial_evaluation_dataset.git`

### 采样规则

**训练集**（`train.json`，2000条）：
- 金融数据 1600 条（80%）：从 Fin-Eva 的 Ant/金融知识（CSV）和 SUFE 的 dev+val（CSV）中随机采样
- 通用数据 400 条（20%）：从 MMLU dev（Parquet）和 C-Eval dev（Parquet）中随机采样
- 随机种子：`seed=42`

**测试集**（3个文件）：
- `mmlu_test.json`（500条）：从 MMLU test 分集中随机采样
- `ceval_test.json`（500条）：从 C-Eval test 分集中随机采样
- `finance_test.json`（1000条）：从 Fin-Eva 金融数据中随机采样（排除训练集已用数据）

### 数据格式

所有数据采用统一的 OpenAI messages 格式，每条数据包含一个 user 消息（问题+选项）和一个 assistant 消息（答案字母）。

### 数据格式转换说明

| 原始格式 | 转换方式 |
|----------|----------|
| Fin-Eva CSV（question, A, B, C, D, answer） | 直接映射为 messages 格式 |
| MMLU Parquet（question, choices[], answer_idx） | answer_idx 为整数索引，通过 `chr(ord('A') + answer_idx)` 转为字母 |
| C-Eval Parquet（question, A, B, C, D, answer） | 直接映射为 messages 格式 |

### dataset_info.json 配置说明

`dataset_info.json` 是 LLaMA-Factory 识别数据集的配置文件，位于 `finetune_data_final/` 目录下。

| 字段 | 值 | 含义 |
|------|-----|------|
| 顶层键 | `finetune_train` | 数据集配置名称，对应 `--dataset` 参数的值 |
| `file_name` | `train.json` | 实际数据文件名，位于 `dataset_dir` 指定的目录下 |
| `formatting` | `sharegpt` | 数据格式类型，LLaMA-Factory 原生支持 alpaca 和 sharegpt 两种 |
| `columns.messages` | `messages` | 指定 JSON 中对话消息列表的字段名 |
| `tags.role_tag` | `role` | 消息中角色字段的键名 |
| `tags.content_tag` | `content` | 消息中内容字段的键名 |
| `tags.user_tag` | `user` | 用户角色的标识值 |
| `tags.assistant_tag` | `assistant` | 助手角色的标识值 |

> LLaMA-Factory 的 `sharegpt` 格式通过 `tags` 配置支持 OpenAI messages 格式，无需额外转换。

## 微调方法

### 使用 LLaMA-Factory Web 端

1. 登录 LLaMA-Factory Web 界面
2. 数据格式选择 **sharegpt**
3. 数据集选择 **finetune_train**
4. 配置微调参数（见下表）

| 参数 | 值 | 含义 |
|------|-----|------|
| 学习率 | 3e-5 | 控制模型参数更新步长，较小的值保护原有知识 |
| 训练轮数 | 2 | 完整遍历训练集的次数，过多会导致过拟合 |
| LoRA Rank | 8 | 低秩分解的秩，控制可训练参数量，值越大表达能力越强但越容易过拟合 |
| LoRA Alpha | 16 | LoRA 缩放因子，通常设为 Rank 的 2 倍，控制 LoRA 更新的强度 |
| 批次大小 | 4 | 每次梯度更新使用的样本数 |
| 梯度累积 | 4 | 累积多个批次的梯度后再更新，等效批次大小 = 4×4 = 16 |
| 输出目录 | `/root/autodl-tmp/LLaMA-Factory/saves/qwen3-4b-finance-final` | 微调权重保存路径 |

### 使用命令行（llamafactory-cli）

#### YAML 配置文件说明

创建 YAML 配置文件（如 `qwen3_finance_sft.yaml`），通过 `llamafactory-cli train qwen3_finance_sft.yaml` 启动训练。

**模型配置**

| 参数 | 值 | 含义 |
|------|-----|------|
| `model_name_or_path` | `/root/autodl-tmp/LLaMA-Factory/saves/qwen3-4b-merged-clean/` | 基础模型路径，即微调的起始模型 |

**微调方法配置**

| 参数 | 值 | 含义 |
|------|-----|------|
| `stage` | `sft` | 训练阶段，sft 表示监督微调（Supervised Fine-Tuning） |
| `do_train` | `true` | 执行训练（设为 false 则只评估） |
| `finetuning_type` | `lora` | 微调方式，lora 为低秩适配，只训练少量参数 |
| `lora_rank` | `8` | LoRA 低秩矩阵的秩，越大可训练参数越多 |
| `lora_alpha` | `16` | LoRA 缩放系数，实际缩放 = alpha / rank |
| `lora_target` | `all` | LoRA 应用的目标模块，all 表示所有线性层 |

**数据集配置**

| 参数 | 值 | 含义 |
|------|-----|------|
| `dataset` | `finetune_train` | 数据集配置名称，对应 dataset_info.json 中的顶层键 |
| `dataset_dir` | `/root/autodl-tmp/flun_train/finetune_data_final` | dataset_info.json 所在目录，LLaMA-Factory 从此目录读取配置和数据文件 |
| `template` | `qwen3` | 对话模板，必须与模型匹配，qwen3 对应 Qwen3 系列的提示词格式 |
| `cutoff_len` | `2048` | 最大序列长度，超过此长度的样本会被截断 |

> **dataset 与 dataset_dir 的关系**：`dataset_dir` 指向包含 `dataset_info.json` 的目录，`dataset` 是该 JSON 文件中的配置名称，LLaMA-Factory 通过配置名称找到对应的 `file_name`（如 `train.json`），再从同一目录加载实际数据文件。

**输出配置**

| 参数 | 值 | 含义 |
|------|-----|------|
| `output_dir` | `/root/autodl-tmp/LLaMA-Factory/saves/qwen3-4b-finance-final` | 微调权重和日志的保存目录 |
| `logging_steps` | `10` | 每 10 步输出一次训练日志 |
| `save_steps` | `100` | 每 100 步保存一次检查点 |

**训练超参数**

| 参数 | 值 | 含义 |
|------|-----|------|
| `per_device_train_batch_size` | `4` | 每个 GPU 的训练批次大小 |
| `gradient_accumulation_steps` | `4` | 梯度累积步数，等效批次 = 4×4 = 16 |
| `learning_rate` | `3.0e-5` | 学习率，控制参数更新幅度 |
| `num_train_epochs` | `2.0` | 训练轮数 |
| `lr_scheduler_type` | `cosine` | 学习率调度策略，cosine 为余弦退火，训练后期学习率逐渐降低 |
| `warmup_ratio` | `0.1` | 预热比例，前 10% 步数学习率从 0 线性增加到设定值 |
| `bf16` | `true` | 使用 BF16 混合精度训练，减少显存占用 |

#### 命令行参数方式

也可以不使用 YAML 文件，直接通过 `llamafactory-cli train` 后跟 `--参数名 值` 的方式传递所有参数，参数名称和含义与 YAML 配置完全一致。

#### 命令行微调注意事项

如果命令行微调静默退出（无报错但未启动），尝试：
- 减小 `per_device_train_batch_size` 到 2 或 1
- 检查 `model_name_or_path` 路径是否正确
- 确认 `dataset_dir` 下存在 `dataset_info.json` 和对应的数据文件

## 合并 LoRA 模型

微调完成后，LoRA 权重是独立于基础模型的适配器，需要合并后才能部署。

使用 `llamafactory-cli export` 命令合并，主要参数说明：

| 参数 | 含义 |
|------|------|
| `model_name_or_path` | 基础模型路径（与微调时相同） |
| `adapter_name_or_path` | LoRA 权重路径（即微调的 output_dir） |
| `template` | 对话模板，与微调时保持一致 |
| `finetuning_type` | 微调类型，与微调时保持一致 |
| `export_dir` | 合并后模型的保存路径 |
| `export_size` | 模型分片大小（GB），设为 2 表示每个文件最大 2GB |
| `export_device` | 合并时使用的设备，cpu 表示在 CPU 上合并，避免 GPU 显存不足 |

## 模型部署

### 使用 vLLM 启动

微调合并后的模型通过 vLLM 部署为 OpenAI 兼容的 API 服务。

启动参数说明：

| 参数 | 含义 |
|------|------|
| 模型路径（位置参数） | 合并后模型的路径 |
| `--served-model-name` | API 中使用的模型名称，调用时通过此名称指定模型 |
| `--max-model-len` | 模型最大上下文长度，影响单次请求能处理的 token 数 |
| `--max-num-seqs` | 单批次最大并发序列数，受 GPU 显存限制 |
| `--port` | API 服务监听端口 |
| `--enforce-eager` | 禁用 CUDA 图优化，使用即时执行模式，提高推理稳定性 |
| `--enable-prefix-caching` | 启用前缀缓存，对相同前缀的请求复用 KV cache，提升推理速度 |

**环境变量说明**：

| 变量 | 含义 |
|------|------|
| `CUDA_LAUNCH_BLOCKING=1` | 强制 GPU 同步执行，禁用异步启动。单 GPU 下同样有效，确保每次推理结果一致，解决测评不稳定问题 |

> **注意**：`CUDA_LAUNCH_BLOCKING=1` 和 `--enforce-eager` 会略微降低推理速度，但能确保结果可复现。生产环境如不需要严格可复现性，可以去掉这两个参数。

## 模型评估

### 使用 evalscope 评估

测试数据已按数据集分为 3 个文件，可分别评估或综合评估。

**evalscope 参数说明**：

| 参数 | 含义 |
|------|------|
| `--model` | 模型名称，需与 vLLM 启动时的 `--served-model-name` 一致 |
| `--api-url` | vLLM 服务的 API 地址 |
| `--api-key` | API 密钥，本地部署填 `EMPTY` 即可 |
| `--eval-batch-size` | 评估时并发请求数，受 vLLM 的 `--max-num-seqs` 限制 |
| `--datasets` | 评估数据集类型，`general_mcq` 表示通用选择题 |
| `--dataset-args` | 数据集参数（JSON 格式），指定本地数据路径和子集列表 |
| `--generation-config` | 生成配置（JSON 格式），`do_sample: false` 和 `temperature: 0` 确保确定性输出 |
| `--seed` | 随机种子，确保评估结果可复现 |

**dataset-args 中 local_path**：指向 `finetune_data_final/` 目录，evalscope 从该目录读取测试数据。

**dataset-args 中 subset_list**：指定要评估的子集，可选值为 `mmlu_test`、`ceval_test`、`finance_test`，可单独指定或组合使用。

### 预期效果

| 指标 | 微调前 | 预期微调后 |
|------|--------|------------|
| MMLU | ~0.786 | ≥0.77（保持稳定） |
| C-Eval | ~0.746 | ≥0.76（小幅提升） |
| Fin-Eva | ~0.778 | ≥0.82（明显提升） |

## 执行步骤总结

1. **准备数据**：下载三个原始数据集，运行 `python generate_data.py` 生成训练集和测试集
2. **上传数据**：将 `finetune_data_final/` 目录复制到服务器
3. **微调模型**：通过 LLaMA-Factory Web 端或命令行执行微调
4. **合并权重**：使用 `llamafactory-cli export` 将 LoRA 权重合并到基础模型
5. **部署模型**：使用 vLLM 启动模型服务（建议启用稳定模式参数）
6. **评估模型**：使用 evalscope 分别评估 MMLU、C-Eval、Fin-Eva 三个测试集

## 注意事项

1. **命令行微调问题**：如果命令行静默退出，尝试减小 `per_device_train_batch_size` 到 2 或 1
2. **数据泄露**：训练集和测试集已确保不重叠
3. **环境变量**：确保设置正确的 `CUDA_VISIBLE_DEVICES` 环境变量
4. **路径适配**：文档中的服务器路径（`/root/autodl-tmp/`）需根据实际环境修改

---

## 许可证

本项目基于 Apache-2.0 许可证。
