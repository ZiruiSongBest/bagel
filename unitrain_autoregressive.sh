#!/bin/bash

# 自回归序列生成训练启动脚本
# 支持多模态自回归训练：文本 -> 图像 -> 文本 -> 图像

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

# 清理GPU缓存
wandb offline
# 设置 CUDA 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 启用CUDA内存池，减少内存碎片  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 强制模型并行（自回归训练暂不支持分布式）
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false

echo "开始自回归序列生成训练..."
echo "当前可用GPU数量: $(nvidia-smi -L | wc -l)"

# 单进程多GPU模型并行训练（自回归训练batch_size固定为1）
python train_autoregressive.py \
    --model_path /remote-home/share/_hf_models/BAGEL-7B-MoT \
    --train_data_path /remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json \
    --batch_size 1 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --output_dir ./outputs/autoregressive_training \
    --logging_steps 1 \
    --save_steps 50 \
    --eval_steps 25 \
    --model_load_mode 1 \
    --fp16 \
    --dataloader_num_workers 0 \
    --text_loss_weight 1.0 \
    --image_loss_weight 1.0 \
    --max_sequence_length 2048

echo "自回归序列生成训练完成！"
