#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bagel

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_unified_generation.py \
       --model_path  /remote-home/share/_hf_models/BAGEL-7B-MoT \
       --train_data_path /remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json \
       --batch_size 1 \
       --gradient_accumulation_steps 32 \
       --num_epochs 3 \
       --learning_rate 1e-5