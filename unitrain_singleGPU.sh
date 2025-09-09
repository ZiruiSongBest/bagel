#!/bin/bash

# 单卡训练启动脚本（修复内存问题版本）
# 适用于单张 A100 GPU 的 BAGEL 模型训练

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

# 设置 CUDA 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 单卡训练
python train_unified_generation.py \
    --model_path /remote-home/share/_hf_models/BAGEL-7B-MoT \
    --train_data_path /remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir ./outputs/singleGPU_training \
    --logging_steps 10 \
    --save_steps 100 \
    --model_load_mode 1

echo "单卡训练完成！"
