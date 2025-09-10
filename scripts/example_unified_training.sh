#!/bin/bash
# 统一训练示例脚本
# 这个脚本展示了如何使用unified_fsdp_trainer.py进行训练

# 设置基本环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_MODE=offline
# 分布式配置
export num_nodes=1
export node_rank=0
export master_addr="localhost"
export master_port="$(python -c 'import socket; s=socket.socket(); s.bind((\"localhost\", 0)); print(s.getsockname()[1]); s.close()')"
export nproc_per_node=3

# 模型和数据路径 - 请根据你的实际路径修改
export model_path="/remote-home/share/_hf_models/BAGEL-7B-MoT"
# export llm_path="models/Qwen2.5-0.5B-Instruct/"
# export vae_path="models/flux/vae/ae.safetensors"
# export vit_path="models/siglip-so400m-14-980-flash-attn2-navit/"

export train_data_path="/remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json"
export val_data_path=""  # 明确设置为空字符串，表示不使用验证数据

# 输出路径
export output_path="results/unified_training_$(date +%Y%m%d_%H%M%S)"
export ckpt_path="$output_path/checkpoints"

# 训练配置
export batch_size=1
export gradient_accumulation_steps=8  # 有效批次大小 = 1 * 8 * 8 GPU = 64
export total_steps=50000
export learning_rate=1e-5
export warmup_steps=1000
export save_every=1000
export log_every=10

# W&B配置
export wandb_project="bagel-unified"
export wandb_name="unified-fsdp-example"
export wandb_runid="1"
export wandb_offline="True"

# FSDP配置
export sharding_strategy="HYBRID_SHARD"
export num_shard=3
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

echo "开始统一训练..."
echo "输出目录: $output_path"

# 调用训练脚本
bash scripts/train_unified_fsdp.sh
