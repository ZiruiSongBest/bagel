# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
cd /workspace/bagel
export PYTHONPATH=/workspace/bagel:$PYTHONPATH
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr="localhost" \
  --master_port=12345 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path /workspace/bagel/models/flux/ae.safetensors \
  --vit_path /workspace/bagel/models/siglip-so400m-14-980-flash-attn2-navit \
  --llm_path /workspace/bagel/models/Qwen2.5-0.5B-Instruct \
  --use_flex True \
  --resume_from "" \
  --wandb_offline True \
  --results_dir /workspace/bagel/results/unified_training_$(date +%Y%m%d_%H%M%S) \
  --checkpoint_dir /workspace/bagel/results/unified_training_$(date +%Y%m%d_%H%M%S)/checkpoints \
  --max_latent_size 64  \
  --num_workers 1 \
  --sharding_strategy FULL_SHARD \
  --num_shard 1 # use small num_workers since the num_used_data (10) are not enough to split