#!/bin/bash

# 多卡分布式训练启动脚本
# 支持 4 张 A100 GPU 的 BAGEL 模型训练

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

# 设置 CUDA 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 分布式训练环境变量
export WORLD_SIZE=4
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# 使用 torchrun 启动多进程分布式训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_unified_generation.py \
    --model_path /remote-home/share/_hf_models/BAGEL-7B-MoT \
    --train_data_path /remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir ./outputs/multiGPU_training \
    --logging_steps 10 \
    --save_steps 100 \
    --model_load_mode 1


