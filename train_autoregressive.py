#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自回归序列生成训练脚本

这个脚本提供了训练支持多模态自回归生成的BAGEL模型的完整流程：
输入: 文本 + 图像
输出: 文本1 -> 图像1 -> 文本2 -> 图像2 (自回归生成)

使用示例:
python train_autoregressive.py \
    --model_path models/BAGEL-7B-MoT \
    --train_data_path data/autoregressive_train.json \
    --output_dir outputs/autoregressive_training \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-5
"""

import os
import sys
import json
import torch
import torch.distributed as dist
import argparse
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from training.autoregressive_trainer import AutoregressiveTrainer, AutoregressiveTrainingConfig
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("不使用分布式训练")
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


def setup_logging(rank=0):
    """设置日志"""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练自回归序列生成模型")
    
    # 模型相关
    parser.add_argument("--model_path", type=str, required=True,
                       help="预训练模型路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点恢复训练")
    
    # 数据相关
    parser.add_argument("--train_data_path", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="验证数据路径")
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                       help="最大序列长度")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小（自回归训练固定为1）")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值")
    
    # 损失权重
    parser.add_argument("--text_loss_weight", type=float, default=1.0,
                       help="文本损失权重")
    parser.add_argument("--image_loss_weight", type=float, default=1.0,
                       help="图像损失权重")
    
    # 保存和日志
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志记录间隔")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存间隔")
    parser.add_argument("--eval_steps", type=int, default=200,
                       help="验证间隔")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="保存的检查点总数限制")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--fp16", action="store_true",
                       help="使用FP16混合精度")
    parser.add_argument("--bf16", action="store_true",
                       help="使用BF16混合精度")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="数据加载器工作进程数")
    
    # 模型加载模式
    parser.add_argument("--model_load_mode", type=int, default=1, choices=[1, 2, 3],
                       help="模型加载模式: 1=标准, 2=NF4量化, 3=INT8量化")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bagel_model(model_path: str, mode: int = 1, use_ddp: bool = False, rank: int = 0):
    """
    加载BAGEL模型（复用原有的加载逻辑）
    """
    if rank == 0:
        logging.info(f"开始加载BAGEL模型，路径: {model_path}")
    
    # 加载配置
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # 初始化模型结构
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # 加载tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 创建transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 设备映射和模型加载
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "30GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    # 确保关键模块在同一设备
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else first_device

    # 加载模型权重
    if mode == 1:
        if rank == 0:
            logging.info("使用标准模式加载模型...")
        
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    else:
        raise NotImplementedError(f"模式 {mode} 未实现")
    
    # 设置为训练模式
    model.train()
    
    if rank == 0:
        logging.info("模型加载完成！")
        logging.info(f"新增的特殊token ID: {new_token_ids}")
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def create_training_config(args) -> AutoregressiveTrainingConfig:
    """创建训练配置"""
    return AutoregressiveTrainingConfig(
        # 数据相关
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        max_sequence_length=args.max_sequence_length,
        
        # 训练相关
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        
        # 优化器相关
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # 损失权重
        text_loss_weight=args.text_loss_weight,
        image_loss_weight=args.image_loss_weight,
        
        # 保存和日志
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        
        # 其他
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
    )


def main():
    """主函数"""
    # 设置分布式训练
    use_ddp, rank, world_size, local_rank = setup_distributed()
    
    # 设置日志
    setup_logging(rank)
    logger = logging.getLogger(__name__)
    
    # 解析参数
    args = parse_args()
    
    # 强制batch_size为1（自回归训练的要求）
    if args.batch_size != 1:
        logger.warning("自回归训练只支持batch_size=1，自动设置为1")
        args.batch_size = 1
    
    # 设置随机种子
    set_seed(args.seed + rank)
    
    # 检查数据路径
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"训练数据路径不存在: {args.train_data_path}")
    
    if args.val_data_path and not os.path.exists(args.val_data_path):
        raise FileNotFoundError(f"验证数据路径不存在: {args.val_data_path}")
    
    # 创建输出目录
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存参数
        with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info("=" * 60)
        logger.info("开始自回归序列生成训练")
        logger.info("=" * 60)
        logger.info(f"使用分布式训练: {use_ddp}")
        if use_ddp:
            logger.info(f"总进程数: {world_size}")
            logger.info(f"当前进程rank: {rank}")
            logger.info(f"本地rank: {local_rank}")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"训练数据: {args.train_data_path}")
        logger.info(f"验证数据: {args.val_data_path}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"批次大小: {args.batch_size}")
        logger.info(f"训练轮数: {args.num_epochs}")
        logger.info(f"学习率: {args.learning_rate}")
        logger.info("=" * 60)
    
    # 同步所有进程
    if use_ddp:
        dist.barrier()
    
    try:
        # 加载模型
        if rank == 0:
            logger.info("正在加载模型...")
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=args.model_path,
            mode=args.model_load_mode,
            use_ddp=use_ddp,
            rank=rank
        )
        
        # 创建训练配置
        training_config = create_training_config(args)
        
        # 创建训练器
        trainer = AutoregressiveTrainer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            config=training_config,
            use_ddp=use_ddp,
            rank=rank,
            world_size=world_size,
        )
        
        # 从检查点恢复（如果指定）
        if args.resume_from_checkpoint and rank == 0:
            logger.info(f"从检查点恢复训练: {args.resume_from_checkpoint}")
            trainer.load_checkpoint(args.resume_from_checkpoint)
        
        # 开始训练
        trainer.train()
        
        if rank == 0:
            logger.info("=" * 60)
            logger.info("自回归序列生成训练完成！")
            logger.info(f"最终模型保存在: {args.output_dir}")
            logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
