# 统一生成模型训练指南

这个目录包含了训练BAGEL统一生成模型（支持文本和图像联合生成）的完整代码。

## 🚀 快速开始

### 1. 准备训练数据

训练数据应该是包含文本和图像混合序列的JSON/JSONL文件。支持多种数据格式：

**对话格式示例**:
```json
{
  "conversations": [
    {
      "role": "user",
      "type": "text",
      "content": "请生成一张猫的图片"
    },
    {
      "role": "assistant", 
      "type": "image",
      "image_path": "/path/to/cat.jpg"
    }
  ]
}
```

**文本到图像格式示例**:
```json
{
  "text_prompt": "A beautiful sunset over the ocean",
  "image_path": "/path/to/sunset.jpg"
}
```

更多数据格式请参考 `sample_data_format.json`。

### 2. 安装依赖

```bash
pip install torch torchvision transformers accelerate
pip install wandb  # 可选，用于实验追踪
pip install nltk rouge-score evaluate  # 可选，用于文本评估
pip install scipy scikit-image  # 可选，用于图像评估
```

### 3. 开始训练

```bash
python train_unified_generation.py \
    --model_path models/BAGEL-7B-MoT \
    --train_data_path data/train.jsonl \
    --val_data_path data/val.jsonl \
    --output_dir outputs/unified_training \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --wandb_project "bagel-unified-training"
```

## 📁 文件结构

```
training/
├── unified_data_processor.py      # 数据处理器
├── unified_trainer.py             # 训练器
├── evaluation_metrics.py          # 评估指标
├── sample_data_format.json        # 数据格式示例
└── README_training.md             # 本文件

train_unified_generation.py        # 主训练脚本
```

## 🔧 主要组件说明

### UnifiedGenerationDataset

数据集类，负责：
- 加载和解析多种格式的训练数据
- 将文本和图像转换为模型可处理的format
- 创建损失计算的掩码

### UnifiedTrainer

训练器类，负责：
- 管理训练循环和优化过程
- 计算联合损失（文本CE + 图像MSE）
- 处理梯度累积和学习率调度
- 保存检查点和日志记录

### 评估指标

支持的评估指标：
- **文本**: BLEU, ROUGE, METEOR, BERTScore
- **图像**: FID, CLIP Score
- **综合**: 多模态一致性评估

## 🎯 训练配置

### 关键超参数

- `batch_size`: 建议使用1，因为序列长度差异很大
- `gradient_accumulation_steps`: 8-16，用于模拟更大的batch size
- `learning_rate`: 1e-5到5e-5，取决于模型大小
- `text_loss_weight`: 文本损失权重（默认1.0）
- `image_loss_weight`: 图像损失权重（默认1.0）

### 学习率调度

支持三种调度器：
- `linear`: 线性衰减
- `cosine`: 余弦衰减（推荐）
- `constant`: 常数学习率

### 混合精度训练

```bash
# 使用FP16
python train_unified_generation.py --fp16 ...

# 使用BF16 (如果硬件支持)
python train_unified_generation.py --bf16 ...
```

## 📊 监控和日志

### W&B集成

```bash
python train_unified_generation.py \
    --wandb_project "my-project" \
    --wandb_run_name "unified-training-v1" \
    ...
```

监控的指标包括：
- 总损失、文本损失、图像损失
- 学习率变化
- 梯度范数
- 验证指标

### 检查点管理

- 自动保存训练检查点
- 支持从检查点恢复训练
- 可配置保存频率和保留数量

```bash
# 从检查点恢复
python train_unified_generation.py \
    --resume_from_checkpoint outputs/unified_training/step_1000 \
    ...
```

## 🔍 训练技巧

### 1. 数据平衡

确保训练数据中文本生成和图像生成的样本数量相对平衡。

### 2. 损失权重调整

根据具体任务调整文本和图像损失的权重：

```bash
# 更重视图像生成质量
python train_unified_generation.py \
    --text_loss_weight 0.5 \
    --image_loss_weight 2.0 \
    ...
```

### 3. 序列长度管理

- 较长的序列会占用更多内存
- 可以考虑截断过长的序列
- 使用梯度累积来模拟更大的batch size

### 4. 验证策略

- 定期在验证集上评估模型
- 使用多种评估指标
- 保存验证效果最好的模型

## 🐛 常见问题

### Q: 内存不足怎么办？

A: 尝试以下方法：
- 减小batch_size到1
- 增加gradient_accumulation_steps
- 使用FP16/BF16混合精度
- 减小max_sequence_length

### Q: 训练速度慢怎么办？

A: 可以尝试：
- 使用更多GPU进行分布式训练
- 优化数据加载（增加num_workers）
- 使用编译优化（torch.compile）

### Q: 如何调试数据处理？

A: 可以：
- 检查sample_data_format.json确认数据格式
- 使用小数据集测试
- 检查日志中的数据统计信息

## 📈 性能优化

### 分布式训练

```bash
# 使用多GPU训练
torchrun --nproc_per_node=4 train_unified_generation.py ...
```

### 内存优化

- 使用量化模型（NF4/INT8）
- 启用gradient checkpointing
- 使用DeepSpeed ZeRO

### 推理优化

训练完成后，可以使用训练好的模型进行推理：

```python
from inferencer import InterleaveInferencer

# 加载训练好的模型
model.load_state_dict(torch.load("path/to/trained_model.pt"))

# 创建推理器
inferencer = InterleaveInferencer(...)

# 进行统一生成
result = inferencer.unified_generate("生成一张猫的图片")
```

## 📚 参考资料

- [BAGEL论文](链接)
- [训练数据准备指南](sample_data_format.json)
- [评估指标说明](evaluation_metrics.py)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进训练代码！

---

**注意**: 这是一个实验性的训练框架，可能需要根据具体需求进行调整。建议在小数据集上先进行测试。
