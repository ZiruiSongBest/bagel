#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成模型的训练器

这个模块实现了支持文本和图像联合生成的训练循环，包括：
1. 联合损失计算（文本CE损失 + 图像MSE损失）
2. 梯度累积和优化
3. 训练状态监控和保存
4. 学习率调度
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

from modeling.bagel import Bagel
from training.unified_data_processor import UnifiedGenerationDataset


@dataclass
class UnifiedTrainingConfig:
    """统一训练配置"""
    
    # 数据相关
    train_data_path: str
    val_data_path: Optional[str] = None
    max_sequence_length: int = 2048
    max_image_tokens: int = 1024
    
    # 训练相关
    batch_size: int = 1  # 建议使用1，因为序列长度差异大
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # 优化器相关
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    warmup_ratio: float = 0.1
    
    # 损失权重
    text_loss_weight: float = 1.0
    image_loss_weight: float = 1.0
    
    # 保存和日志
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    save_epochs: int = 1  # 每多少个epoch保存一次，1表示每个epoch都保存
    eval_steps: int = 200
    save_total_limit: int = 3
    
    # 分布式训练
    local_rank: int = -1
    deepspeed_config: Optional[str] = None
    
    # 实验跟踪
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # 其他
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    
    def to_dict(self):
        return asdict(self)


class UnifiedTrainer:
    """统一生成模型训练器"""
    
    def __init__(
        self,
        model: Bagel,
        vae_model: nn.Module,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids: Dict[str, int],
        config: UnifiedTrainingConfig,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.config = config
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        
        # 设置设备
        if use_ddp:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # VAE模型始终移动到当前设备
        if hasattr(self.vae_model, 'to'):
            self.vae_model.to(self.device)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        
        # 损失函数
        self.text_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.image_criterion = nn.MSELoss()
        
        # 实验跟踪
        self.wandb_initialized = False
        if config.wandb_project:
            self._init_wandb()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_wandb(self):
        """初始化W&B"""
        try:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
            self.wandb_initialized = True
            self.logger.info("W&B initialized successfully")
        except Exception as e:
            self.wandb_initialized = False
            self.logger.warning(f"Failed to initialize W&B: {e}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分离需要权重衰减和不需要权重衰减的参数
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "LayerNorm.weight", "layernorm.weight"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        return optimizer
    
    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        # 这里需要根据总训练步数来创建调度器
        # 在get_train_dataloader()之后再调用
        return None
    
    def get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        from training.unified_data_processor import create_unified_dataloader
        from torch.utils.data import DistributedSampler
        
        train_dataset = UnifiedGenerationDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
            max_sequence_length=self.config.max_sequence_length,
            max_image_tokens=self.config.max_image_tokens,
        )
        
        # 分布式采样器
        sampler = None
        shuffle = True
        if self.use_ddp:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            shuffle = False  # 使用sampler时不能设置shuffle
        
        return create_unified_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
        )
    
    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """获取验证数据加载器"""
        if not self.config.val_data_path:
            return None
            
        from training.unified_data_processor import create_unified_dataloader
        from torch.utils.data import DistributedSampler
        
        val_dataset = UnifiedGenerationDataset(
            data_path=self.config.val_data_path,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
            max_sequence_length=self.config.max_sequence_length,
            max_image_tokens=self.config.max_image_tokens,
        )
        
        # 分布式采样器（验证时不打乱）
        sampler = None
        if self.use_ddp:
            sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        
        return create_unified_dataloader(
            val_dataset,
            batch_size=1,  # 验证时使用batch_size=1
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
        )
    
    def train(self):
        """主训练循环"""
        if self.rank == 0:
            self.logger.info("开始训练...")
        
        # 获取数据加载器
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        
        # 计算总训练步数并创建学习率调度器
        total_steps = len(train_dataloader) * self.config.num_epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)
        
        self.lr_scheduler = self._create_lr_scheduler_with_steps(total_steps)
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # 分布式训练时设置sampler的epoch
            if self.use_ddp and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            if self.rank == 0:
                self.logger.info(f"开始训练 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = self._train_epoch(train_dataloader)
            
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch + 1} 训练损失: {epoch_loss:.4f}")
            
            # 验证
            if eval_dataloader and self.rank == 0:  # 只在主进程进行验证
                val_loss = self._evaluate(eval_dataloader)
                self.logger.info(f"Epoch {epoch + 1} 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model")
            
            # 保存检查点（只在主进程）- 根据save_epochs间隔保存
            if self.rank == 0 and (epoch + 1) % self.config.save_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")
            
            # 同步所有进程
            if self.use_ddp:
                import torch.distributed as dist
                dist.barrier()
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        if self.rank == 0:
            self.logger.info("训练完成!")
    
    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_text_loss = 0.0
        total_image_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            loss_dict = self._training_step(batch)
            
            total_loss += loss_dict['total_loss']
            total_text_loss += loss_dict['text_loss']
            total_image_loss += loss_dict['image_loss']
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss']:.4f}",
                'Text': f"{loss_dict['text_loss']:.4f}",
                'Image': f"{loss_dict['image_loss']:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 日志记录
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(loss_dict)
            
            # 验证
            if (self.config.eval_steps > 0 and 
                self.global_step % self.config.eval_steps == 0 and
                self.global_step > 0):
                eval_dataloader = self.get_eval_dataloader()
                if eval_dataloader:
                    val_loss = self._evaluate(eval_dataloader)
                    self.logger.info(f"Step {self.global_step} 验证损失: {val_loss:.4f}")
            
            # 保存检查点
            if (self.config.save_steps > 0 and 
                self.global_step % self.config.save_steps == 0 and
                self.global_step > 0):
                self._save_checkpoint(f"step_{self.global_step}")
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        return total_loss / len(train_dataloader)
    
    def _training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单个训练步骤"""
        # 将数据移到设备上
        if self.use_ddp:
            # DDP模式：使用当前设备
            device = self.device
        else:
            # 非DDP模式：使用模型的主设备
            device = next(self.model.parameters()).device
        
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # 准备模型输入
        model_inputs = self._prepare_model_inputs(batch)
        
        # 前向传播（禁用混合精度以避免数据类型问题）
        # 由于模型使用BFloat16权重，但某些操作强制转换为Float32，
        # 导致数据类型不匹配，这里禁用autocast让PyTorch自动处理类型转换
        outputs = self.model(**model_inputs)
        
        # 计算损失
        loss_dict = self._compute_loss(outputs, batch)
        
        # 反向传播
        total_loss = loss_dict['total_loss']
        if self.config.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.config.gradient_accumulation_steps
        
        total_loss.backward()
        
        # 梯度累积和优化
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            # 优化器步骤
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'text_loss': loss_dict['text_loss'],
            'image_loss': loss_dict['image_loss'],
        }
    
    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """准备模型输入"""
        # 获取模型的主设备（对于模型并行很重要）
        if self.use_ddp:
            model_device = self.device
        else:
            model_device = next(self.model.parameters()).device
            
        model_inputs = {
            'sequence_length': batch['sequence_length'],
            'packed_text_ids': batch['packed_text_ids'],
            'packed_text_indexes': batch['packed_text_indexes'],
            'sample_lens': batch.get('sample_lens', [batch['sequence_length']]),
            'packed_position_ids': batch['packed_position_ids'],
            # 添加注意力掩码相关参数
            'split_lens': batch.get('split_lens', [batch['sequence_length']]),
            'attn_modes': batch.get('attn_modes', ['causal']),
        }
        
        # 如果有嵌套注意力掩码，则直接使用它们
        if 'nested_attention_masks' in batch:
            model_inputs['nested_attention_masks'] = batch['nested_attention_masks']
        
        # 添加VIT输入（如果存在）
        if 'packed_vit_tokens' in batch:
            # 确保VIT tokens与模型权重类型匹配
            vit_tokens = batch['packed_vit_tokens']
            if vit_tokens.dtype == torch.float32:
                vit_tokens = vit_tokens.to(torch.bfloat16)
            
            # 确保所有索引张量都在正确的设备上
            vit_token_indexes = batch['packed_vit_token_indexes']
            if hasattr(vit_token_indexes, 'device') and vit_token_indexes.device != model_device:
                vit_token_indexes = vit_token_indexes.to(model_device)
                
            model_inputs.update({
                'packed_vit_tokens': vit_tokens,
                'packed_vit_token_indexes': vit_token_indexes,
                'packed_vit_position_ids': batch['packed_vit_position_ids'],
                'vit_token_seqlens': batch['vit_token_seqlens'],
            })
        
        # 添加VAE输入（如果存在）
        if 'padded_vae_images' in batch:
            # 使用VAE编码图像
            with torch.no_grad():
                padded_latent = self.vae_model.encode(batch['padded_vae_images'])
            
            # 确保VAE latent数据类型与模型权重匹配
            if padded_latent.dtype == torch.float32:
                padded_latent = padded_latent.to(torch.bfloat16)
            
            # 确保timesteps数据类型与模型权重匹配
            timesteps = batch['packed_timesteps']
            if timesteps.dtype == torch.float32:
                timesteps = timesteps.to(torch.bfloat16)
            
            # 确保VAE索引张量都在正确的设备上
            vae_token_indexes = batch['packed_vae_token_indexes']
            if hasattr(vae_token_indexes, 'device') and vae_token_indexes.device != model_device:
                vae_token_indexes = vae_token_indexes.to(model_device)
            
            model_inputs.update({
                'padded_latent': padded_latent,
                'patchified_vae_latent_shapes': batch['patchified_vae_latent_shapes'],
                'packed_latent_position_ids': batch['packed_vae_position_ids'],
                'packed_vae_token_indexes': vae_token_indexes,
                'packed_timesteps': timesteps,
            })
        
        # 添加损失掩码 - 确保所有损失相关的张量都在正确设备上
        ce_loss_indexes = batch['text_loss_mask']
        mse_loss_indexes = batch['image_loss_mask']
        
        if hasattr(ce_loss_indexes, 'device') and ce_loss_indexes.device != model_device:
            ce_loss_indexes = ce_loss_indexes.to(model_device)
        if hasattr(mse_loss_indexes, 'device') and mse_loss_indexes.device != model_device:
            mse_loss_indexes = mse_loss_indexes.to(model_device)
            
        # 处理标签数据 - 修复数据格式不兼容问题
        if 'packed_label_ids' in batch:
            # 如果batch中有专门的packed_label_ids，直接使用
            packed_label_ids = batch['packed_label_ids']
        else:
            # 统一数据处理器使用text_loss_mask而不是原始的ce_loss_indexes格式
            # 我们需要将text_loss_mask转换为与原始PackedDataset兼容的格式
            packed_text_ids = batch['packed_text_ids']
            
            # 对于统一数据处理器，我们直接使用packed_text_ids作为标签
            # 但需要确保与ce_loss_indexes长度匹配
            if len(packed_text_ids) == ce_loss_indexes.sum().item():
                # 如果长度匹配，直接使用
                packed_label_ids = packed_text_ids
            else:
                # 如果长度不匹配，说明ce_loss_indexes是掩码形式
                # 创建与ce_loss_indexes长度相同的标签序列，然后提取需要的部分
                device = packed_text_ids.device
                # 创建一个与sequence_length相同长度的标签序列
                full_sequence_length = len(ce_loss_indexes)
                if len(packed_text_ids) < full_sequence_length:
                    # 如果packed_text_ids较短，用padding扩展
                    padding_length = full_sequence_length - len(packed_text_ids)
                    pad_token = packed_text_ids[0]  # 使用第一个token作为padding
                    padding = torch.full((padding_length,), pad_token, device=device, dtype=packed_text_ids.dtype)
                    full_labels = torch.cat([packed_text_ids, padding])
                else:
                    # 如果packed_text_ids较长，截断到合适长度
                    full_labels = packed_text_ids[:full_sequence_length]
                
                # 只取ce_loss_indexes为True的位置作为标签
                packed_label_ids = full_labels[ce_loss_indexes]
        
        model_inputs.update({
            'ce_loss_indexes': ce_loss_indexes,
            'mse_loss_indexes': mse_loss_indexes,
            'packed_label_ids': packed_label_ids,
        })
        
        return model_inputs
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, float]:
        """计算联合损失"""
        text_loss = 0.0
        image_loss = 0.0
        
        # 文本损失 (Cross Entropy)
        if outputs.get('ce') is not None and len(outputs['ce']) > 0:
            text_loss = outputs['ce'].mean()
        
        # 图像损失 (MSE)
        if outputs.get('mse') is not None and len(outputs['mse']) > 0:
            image_loss = outputs['mse'].mean()
        
        # 总损失
        total_loss = (
            self.config.text_loss_weight * text_loss + 
            self.config.image_loss_weight * image_loss
        )
        
        return {
            'total_loss': total_loss,
            'text_loss': text_loss.item() if isinstance(text_loss, torch.Tensor) else text_loss,
            'image_loss': image_loss.item() if isinstance(image_loss, torch.Tensor) else image_loss,
        }
    
    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="验证中"):
                main_device = next(self.model.parameters()).device
                batch = {k: v.to(main_device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                model_inputs = self._prepare_model_inputs(batch)
                
                with torch.autocast(device_type="cuda", enabled=self.config.fp16, dtype=torch.bfloat16):
                    outputs = self.model(**model_inputs)
                
                loss_dict = self._compute_loss(outputs, batch)
                total_loss += loss_dict['total_loss']
        
        self.model.train()
        return total_loss / len(eval_dataloader)
    
    def _log_metrics(self, loss_dict: Dict[str, float]):
        """记录训练指标"""
        metrics = {
            'train/total_loss': loss_dict['total_loss'],
            'train/text_loss': loss_dict['text_loss'],
            'train/image_loss': loss_dict['image_loss'],
            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
            'train/global_step': self.global_step,
            'train/epoch': self.epoch,
        }
        
        if self.wandb_initialized and hasattr(wandb, 'log'):
            wandb.log(metrics)
        
        self.logger.info(
            f"Step {self.global_step}: "
            f"Loss={loss_dict['total_loss']:.4f}, "
            f"Text={loss_dict['text_loss']:.4f}, "
            f"Image={loss_dict['image_loss']:.4f}"
        )
    
    def _save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        if self.rank != 0:  # 只在主进程保存
            return
            
        checkpoint_dir = Path(self.config.output_dir) / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存模型（如果是DDP，需要保存原始模型）
        if self.use_ddp:
            torch.save(self.model.module.state_dict(), checkpoint_dir / "model.pt")
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # 保存优化器和调度器状态
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_dir / "trainer_state.pt")
        
        # 保存配置
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"检查点已保存到 {checkpoint_dir}")
        
        # 清理旧检查点
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        if self.config.save_total_limit <= 0:
            return
        
        output_dir = Path(self.config.output_dir)
        checkpoints = [d for d in output_dir.iterdir() 
                      if d.is_dir() and d.name.startswith(('step_', 'epoch_'))]
        
        if len(checkpoints) <= self.config.save_total_limit:
            return
        
        # 按修改时间排序，删除最旧的
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        for checkpoint in checkpoints[:-self.config.save_total_limit]:
            import shutil
            shutil.rmtree(checkpoint)
            self.logger.info(f"删除旧检查点: {checkpoint}")
    
    def _create_lr_scheduler_with_steps(self, total_steps: int):
        """创建学习率调度器（知道总步数后）"""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # 常数学习率
            return None
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_path)
        
        # 加载模型
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"模型已从 {model_path} 加载")
        
        # 加载训练状态
        trainer_state_path = checkpoint_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer'])
            if self.lr_scheduler and state.get('lr_scheduler'):
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
            self.logger.info(f"训练状态已从 {trainer_state_path} 加载")


if __name__ == "__main__":
    # 测试训练器配置
    config = UnifiedTrainingConfig(
        train_data_path="/path/to/train_data",
        val_data_path="/path/to/val_data",
        output_dir="./outputs/unified_training",
        batch_size=1,
        gradient_accumulation_steps=8,
        num_epochs=3,
        learning_rate=1e-5,
    )
    
    print("统一训练配置:")
    print(json.dumps(config.to_dict(), indent=2))
