#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成模型训练脚本

这个脚本提供了训练支持文本和图像联合生成的BAGEL模型的完整流程，包括：
1. 模型和数据加载
2. 训练配置管理
3. 分布式训练支持
4. 检查点管理
5. 实验追踪

使用示例:
python train_unified_generation.py \
    --model_path models/BAGEL-7B-MoT \
    --train_data_path data/unified_train.jsonl \
    --val_data_path data/unified_val.jsonl \
    --output_dir outputs/unified_training \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 1e-5
"""

import os
import sys
import json
import torch
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
from training.unified_trainer import UnifiedTrainer, UnifiedTrainingConfig
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练统一生成模型")
    
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
    parser.add_argument("--max_image_tokens", type=int, default=1024,
                       help="单个图像的最大token数")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
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
    
    # 学习率调度
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="热身比例")
    
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
    
    # 实验追踪
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B运行名称")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--fp16", action="store_true",
                       help="使用FP16混合精度")
    parser.add_argument("--bf16", action="store_true",
                       help="使用BF16混合精度")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
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


def load_bagel_model(model_path: str, mode: int = 1):
    """
    加载BAGEL模型
    
    Args:
        model_path: 模型路径
        mode: 加载模式 (1=normal, 2=NF4, 3=INT8)
    
    Returns:
        tuple: (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """
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
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
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

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # 根据模式加载模型权重
    if mode == 1:
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
    elif mode == 2:  # NF4
        logging.info("使用NF4量化模式加载模型...")
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=False, 
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:  # INT8
        logging.info("使用INT8量化模式加载模型...")
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, 
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"模式 {mode} 未实现")
    
    # 设置为训练模式
    model.train()
    
    logging.info("模型加载完成！")
    logging.info(f"新增的特殊token ID: {new_token_ids}")
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def create_training_config(args) -> UnifiedTrainingConfig:
    """创建训练配置"""
    return UnifiedTrainingConfig(
        # 数据相关
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        max_sequence_length=args.max_sequence_length,
        max_image_tokens=args.max_image_tokens,
        
        # 训练相关
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        
        # 优化器相关
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # 学习率调度
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        
        # 损失权重
        text_loss_weight=args.text_loss_weight,
        image_loss_weight=args.image_loss_weight,
        
        # 保存和日志
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        
        # 实验跟踪
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        
        # 其他
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
    )


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查数据路径
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"训练数据路径不存在: {args.train_data_path}")
    
    if args.val_data_path and not os.path.exists(args.val_data_path):
        raise FileNotFoundError(f"验证数据路径不存在: {args.val_data_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 60)
    logger.info("开始统一生成模型训练")
    logger.info("=" * 60)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"训练数据: {args.train_data_path}")
    logger.info(f"验证数据: {args.val_data_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"梯度累积步数: {args.gradient_accumulation_steps}")
    logger.info(f"训练轮数: {args.num_epochs}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info("=" * 60)
    
    try:
        # 加载模型
        logger.info("正在加载模型...")
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=args.model_path,
            mode=args.model_load_mode
        )
        
        # 创建训练配置
        training_config = create_training_config(args)
        
        # 创建训练器
        trainer = UnifiedTrainer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            config=training_config,
        )
        
        # 从检查点恢复（如果指定）
        if args.resume_from_checkpoint:
            logger.info(f"从检查点恢复训练: {args.resume_from_checkpoint}")
            trainer.load_checkpoint(args.resume_from_checkpoint)
        
        # 开始训练
        trainer.train()
        
        logger.info("=" * 60)
        logger.info("训练完成！")
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
