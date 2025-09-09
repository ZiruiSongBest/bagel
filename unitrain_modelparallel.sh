#!/bin/bash

# 模型并行训练启动脚本
# 适用于大模型的跨GPU模型并行训练

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

# 清理GPU缓存
wandb offline
# 设置 CUDA 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 启用CUDA内存池，减少内存碎片  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 强制模型并行
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false

echo "开始模型并行训练..."
echo "当前可用GPU数量: $(nvidia-smi -L | wc -l)"

# 单进程多GPU模型并行训练
python train_unified_generation.py \
    --model_path /remote-home/share/_hf_models/BAGEL-7B-MoT \
    --train_data_path /remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir ./outputs/modelparallel_training \
    --logging_steps 10 \
    --save_steps 100 \
    --save_epochs 50 \
    --model_load_mode 1 \
    --fp16 \
    --dataloader_num_workers 0 \
    --wandb_project "Uni" \
    --wandb_run_name "modelparallel-training-$(date +%Y%m%d-%H%M%S)"

