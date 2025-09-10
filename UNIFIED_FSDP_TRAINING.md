# 统一训练FSDP框架使用指南

## 概述

这个文档介绍如何使用新的统一训练FSDP框架，它结合了：
- 你的 `UnifiedGenerationDataset` 统一训练数据处理
- `pretrain_unified_navit.py` 成熟的FSDP训练基础设施

## 主要优势

✅ **完整的FSDP支持** - 复用成熟的分布式训练框架  
✅ **EMA (指数移动平均)** - 自动维护模型权重的EMA版本  
✅ **检查点管理** - 完善的训练状态保存和恢复  
✅ **梯度累积** - 支持大批次训练  
✅ **W&B集成** - 完整的实验追踪  
✅ **内存优化** - 激活检查点、CPU offload等  

## 文件结构

```
train/
├── unified_fsdp_trainer.py          # 新的统一FSDP训练器
├── pretrain_unified_navit.py        # 原始FSDP训练框架
└── train_utils.py                   # 训练工具函数

scripts/
├── train_unified_fsdp.sh            # 统一训练启动脚本
├── example_unified_training.sh      # 使用示例
└── train.sh                         # 原始训练脚本

training/
├── unified_trainer.py               # 你的原始统一训练器
└── unified_data_processor.py        # 统一数据处理器
```

## 快速开始

### 1. 准备数据

确保你的统一训练数据已经按照 `UnifiedGenerationDataset` 的格式准备好：

```bash
# 训练数据格式 (JSONL)
data/unified_train.jsonl
data/unified_val.jsonl  # 可选，如果没有验证数据可以不提供
```

**注意**: 验证数据是可选的。如果你没有验证数据，只需要确保 `val_data_path` 为空字符串即可。

### 2. 配置模型路径

修改 `scripts/example_unified_training.sh` 中的路径：

```bash
export model_path="你的/BAGEL-7B-MoT/路径"
export llm_path="你的/Qwen2.5-0.5B-Instruct/路径"
export vae_path="你的/flux/vae/ae.safetensors/路径"
export vit_path="你的/siglip-so400m-14-980-flash-attn2-navit/路径"

export train_data_path="你的/unified_train.jsonl/路径"
export val_data_path="你的/unified_val.jsonl/路径"
```

### 3. 启动训练

```bash
# 使用示例配置训练
bash scripts/example_unified_training.sh

# 或者直接使用训练脚本
bash scripts/train_unified_fsdp.sh
```

## 详细配置

### 分布式训练配置

```bash
export num_nodes=1                    # 节点数量
export node_rank=0                    # 当前节点rank
export master_addr="localhost"        # 主节点地址
export master_port="12345"           # 主节点端口
export nproc_per_node=8               # 每节点GPU数
```

### 训练超参数

```bash
export batch_size=1                   # 批次大小 (建议保持1)
export gradient_accumulation_steps=8  # 梯度累积步数
export total_steps=50000             # 总训练步数
export learning_rate=1e-5            # 学习率
export warmup_steps=1000             # 热身步数
export save_every=1000               # 保存间隔
export log_every=10                  # 日志间隔
```

### FSDP配置

```bash
export sharding_strategy="HYBRID_SHARD"  # 分片策略
export num_shard=8                       # 分片数量
export cpu_offload="False"              # CPU卸载
```

### 损失权重

```bash
export text_loss_weight=1.0          # 文本损失权重
export image_loss_weight=1.0         # 图像损失权重
```

### 模型冻结

```bash
export freeze_vae="True"             # 冻结VAE (推荐)
export freeze_llm="False"            # 是否冻结语言模型
export freeze_vit="False"            # 是否冻结视觉编码器
```

## 主要特性

### 1. 数据适配器

`UnifiedDatasetWrapper` 类负责将你的 `UnifiedGenerationDataset` 适配到原有的训练循环：

- 自动处理批次格式转换
- 兼容原有的损失计算逻辑
- 支持检查点恢复的数据状态追踪

### 2. 梯度累积

支持梯度累积来模拟大批次训练：

```python
# 有效批次大小 = batch_size × gradient_accumulation_steps × num_gpus
# 例如: 1 × 8 × 8 = 64
```

### 3. 统一损失计算

自动计算和组合文本和图像损失：

```python
total_loss = text_loss * text_loss_weight + image_loss * image_loss_weight
```

### 4. 内存优化

- **激活检查点**: 自动应用激活检查点来节省显存
- **FSDP分片**: 将模型参数分片到多个GPU
- **CPU卸载**: 可选的参数CPU卸载

## 与原系统的对比

| 特性 | 原始 unified_trainer.py | 新的 unified_fsdp_trainer.py |
|------|------------------------|------------------------------|
| FSDP支持 | ❌ | ✅ 完整支持 |
| EMA | ❌ | ✅ 自动维护 |
| 检查点管理 | 基础 | ✅ 完善的状态管理 |
| 内存优化 | 有限 | ✅ 多种优化策略 |
| 分布式训练 | 基础DDP | ✅ 高级FSDP |
| 梯度累积 | ✅ | ✅ |
| W&B集成 | ✅ | ✅ |

## 监控和调试

### W&B监控

训练会自动记录以下指标：
- `ce`: 文本交叉熵损失
- `mse`: 图像MSE损失  
- `lr`: 学习率
- `mem_allocated`: GPU内存使用
- `total_samples`: 处理的样本数

### 检查点

训练会定期保存检查点到 `checkpoint_dir`：
- `step_XXX/`: 包含模型、优化器、调度器状态
- `ema_state.pt`: EMA模型权重
- `train_state.pt`: 训练状态信息

### 恢复训练

```bash
# 从特定检查点恢复
export resume_from="results/checkpoints/step_5000"

# 自动从最新检查点恢复
export auto_resume="True"
```

## 故障排除

### 1. 内存不足

尝试以下优化：
```bash
export cpu_offload="True"           # 启用CPU卸载
export batch_size=1                 # 减小批次大小
export gradient_accumulation_steps=16  # 增加梯度累积
```

### 2. 数据格式错误

确保 `UnifiedGenerationDataset` 输出包含：
- `input_ids`: 输入token ID
- `attention_mask`: 注意力掩码
- `labels`: 标签 (用于CE损失)
- `padded_images`: 图像数据 (如果有)

### 3. 训练不收敛

调整超参数：
```bash
export learning_rate=5e-6           # 降低学习率
export warmup_steps=2000            # 增加热身步数
export text_loss_weight=2.0         # 调整损失权重
export image_loss_weight=0.5
```

## 性能建议

1. **批次大小**: 建议保持 `batch_size=1`，通过 `gradient_accumulation_steps` 来增加有效批次大小
2. **FSDP策略**: `HYBRID_SHARD` 通常提供最好的性能
3. **内存管理**: 对于76GB+的模型，建议启用激活检查点
4. **数据加载**: 统一数据集较复杂，建议 `num_workers=1`

## 总结

新的统一FSDP训练框架让你能够：
- 保留你的统一训练数据处理逻辑
- 利用成熟的FSDP训练基础设施
- 获得更好的内存效率和训练稳定性
- 享受完整的实验追踪和检查点管理

这样你就可以专注于数据和模型逻辑，而不用担心分布式训练的复杂性！
