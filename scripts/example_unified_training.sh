#!/bin/bash
# 统一训练示例脚本
# 这个脚本展示了如何使用unified_fsdp_trainer.py进行训练

# 设置基本环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用两个GPU
export WANDB_MODE=offline
# 分布式配置
export num_nodes=1
export node_rank=0
export master_addr="localhost"
export master_port=12345
export nproc_per_node=2  # 两个GPU

# 模型和数据路径 - 请根据你的实际路径修改
# export model_path="/workspace/bagel/models"


export llm_path="/workspace/bagel/models/Qwen2.5-0.5B-Instruct"
export vae_path="/workspace/bagel/models/flux/ae.safetensors"
export vit_path="/workspace/bagel/models/siglip-so400m-14-980-flash-attn2-navit"

export train_data_path="/workspace/bagel/dataset/demo/demo_sample/anno.json"
export val_data_path=""  # 明确设置为空字符串，表示不使用验证数据

# 输出路径
export output_path="results/unified_training_$(date +%Y%m%d_%H%M%S)"
export ckpt_path="$output_path/checkpoints"

# 训练配置 - 过拟合20个样本100次
export batch_size=1
export gradient_accumulation_steps=1   # 每个optimizer step处理: 1 * 1 = 1个样本  
export total_steps=1000                # 20个样本，每epoch需要20步，100 epoch需要2000步
export learning_rate=1e-4              # 提高学习率以便更快过拟合
export warmup_steps=10                 # 减少warmup步数
export save_every=200                   # 每200步保存一次（约10个epoch）
export log_every=10                     # 更频繁的日志记录

# W&B配置
export wandb_project="bagel-unified"
export wandb_name="overfitting-experiment-20samples"  # 更名以反映过拟合实验
export wandb_runid="1"
export wandb_offline="True"

# FSDP配置 - 针对2个GPU优化
export sharding_strategy="FULL_SHARD"  # 完全分片，适合多GPU训练
export num_shard=2
export cpu_offload="False"

# 模型配置
export max_latent_size=64
export max_sequence_length=2048
export max_image_tokens=1024

# 冻结配置
export freeze_vae="True"    # VAE通常保持冻结
export freeze_llm="False"   # 训练语言模型
export freeze_vit="False"   # 训练视觉编码器

# 损失权重
export text_loss_weight=1.0
export image_loss_weight=1.0

echo "开始过拟合训练实验..."
echo "训练样本数: 20"
echo "计划过拟合次数: 100个epoch"
echo "总训练步数(optimizer steps): $total_steps"
echo "每个optimizer step处理样本数: $(($batch_size * $gradient_accumulation_steps))"
echo "每个epoch需要步数: 20步 (每个样本一步)"
echo "预计checkpoint保存次数: $((total_steps / save_every))"
echo "输出目录: $output_path"

# 调用训练脚本
bash scripts/train_unified_fsdp.sh
