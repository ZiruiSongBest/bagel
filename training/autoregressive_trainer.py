#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自回归序列生成训练器

支持多模态自回归训练：文本 -> 图像 -> 文本 -> 图像
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from dataclasses import dataclass

from .autoregressive_data_processor import AutoregressiveDataset, collate_autoregressive_batch


@dataclass
class AutoregressiveTrainingConfig:
    """自回归训练配置"""
    # 数据相关
    train_data_path: str
    val_data_path: Optional[str] = None
    max_sequence_length: int = 2048
    
    # 训练相关
    batch_size: int = 1  # 自回归训练只支持batch_size=1
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # 优化器相关
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 损失权重
    text_loss_weight: float = 1.0
    image_loss_weight: float = 1.0
    
    # 保存和日志
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 200
    save_total_limit: int = 3
    
    # 其他
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0  # 自回归训练建议使用0


class AutoregressiveTrainer:
    """自回归序列生成训练器"""
    
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids: Dict[str, int],
        config: AutoregressiveTrainingConfig,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        初始化训练器
        
        Args:
            model: BAGEL模型
            vae_model: VAE模型
            tokenizer: 分词器
            vae_transform: VAE图像变换
            vit_transform: VIT图像变换
            new_token_ids: 特殊token映射
            config: 训练配置
            use_ddp: 是否使用分布式训练
            rank: 进程rank
            world_size: 总进程数
        """
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
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._setup_datasets()
        self._setup_optimizer()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
    def _setup_datasets(self):
        """设置数据集"""
        self.logger.info("正在设置数据集...")
        
        # 训练数据集
        self.train_dataset = AutoregressiveDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
            max_sequence_length=self.config.max_sequence_length,
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=collate_autoregressive_batch,
            pin_memory=True,
            drop_last=True,
        )
        
        # 验证数据集（如果提供）
        if self.config.val_data_path:
            self.val_dataset = AutoregressiveDataset(
                data_path=self.config.val_data_path,
                tokenizer=self.tokenizer,
                vae_transform=self.vae_transform,
                vit_transform=self.vit_transform,
                new_token_ids=self.new_token_ids,
                max_sequence_length=self.config.max_sequence_length,
            )
            
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=collate_autoregressive_batch,
                pin_memory=True,
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 只对需要梯度的参数进行优化
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.logger.info(f"优化器设置完成，可训练参数数量: {len(trainable_params)}")
    
    def train(self):
        """开始训练"""
        self.logger.info("开始自回归序列生成训练...")
        
        # 设置模型为训练模式
        self.model.train()
        self.vae_model.eval()  # VAE模型保持eval模式
        
        total_steps = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"开始第 {epoch + 1}/{self.config.num_epochs} 个epoch")
            
            for step, batch in enumerate(self.train_dataloader):
                # 检查是否达到最大步数
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    self.logger.info("达到最大训练步数，停止训练")
                    return
                
                # 训练一步
                loss_dict = self._train_step(batch)
                
                # 记录日志
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_progress(loss_dict)
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # 验证
                if (self.val_dataloader and 
                    self.global_step % self.config.eval_steps == 0):
                    self._evaluate()
                
                self.global_step += 1
                total_steps += 1
        
        self.logger.info("训练完成！")
        
        # 保存最终模型
        self._save_checkpoint(is_final=True)
    
    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """执行一个训练步骤"""
        
        # 移动数据到设备
        input_text = batch['input_text']
        input_image = batch['input_image'].to(self.device)
        target_texts = batch['target_texts']
        target_images = [img.to(self.device) for img in batch['target_images']]
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        with torch.amp.autocast("cuda", enabled=self.config.fp16 or self.config.bf16):
            loss_dict = self.model.forward_autoregressive_training(
                input_text=input_text,
                input_image=input_image,
                target_texts=target_texts,
                target_images=target_images,
                tokenizer=self.tokenizer,
                vae_model=self.vae_model,
            )
        
        # 计算总损失
        text_loss = loss_dict["text_loss"]
        image_loss = loss_dict["image_loss"]
        
        total_loss = (text_loss * self.config.text_loss_weight + 
                     image_loss * self.config.image_loss_weight)
        
        # 反向传播
        if self.config.fp16 or self.config.bf16:
            # 使用混合精度训练
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        # 返回损失信息
        return {
            "total_loss": total_loss.detach(),
            "text_loss": text_loss.detach(),
            "image_loss": image_loss.detach(),
        }
    
    def _log_training_progress(self, loss_dict: Dict[str, torch.Tensor]):
        """记录训练进度"""
        if self.rank == 0:  # 只在主进程记录
            message = f"Step {self.global_step:06d} | Epoch {self.epoch:03d} | "
            for key, value in loss_dict.items():
                message += f"{key}: {value.item():.6f} | "
            
            self.logger.info(message)
    
    def _save_checkpoint(self, is_final: bool = False):
        """保存检查点"""
        if self.rank != 0:  # 只在主进程保存
            return
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if is_final:
            checkpoint_path = os.path.join(self.config.output_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点: {checkpoint_path}")
        
        # 清理旧检查点
        if not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        checkpoint_files = []
        for filename in os.listdir(self.config.output_dir):
            if filename.startswith("checkpoint-") and filename.endswith(".pt"):
                checkpoint_files.append(filename)
        
        # 按步数排序
        checkpoint_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        
        # 删除超出限制的旧检查点
        while len(checkpoint_files) > self.config.save_total_limit:
            old_checkpoint = checkpoint_files.pop(0)
            old_path = os.path.join(self.config.output_dir, old_checkpoint)
            os.remove(old_path)
            self.logger.info(f"删除旧检查点: {old_path}")
    
    def _evaluate(self):
        """执行验证"""
        if not self.val_dataloader:
            return
        
        self.logger.info("开始验证...")
        self.model.eval()
        
        total_losses = {"text_loss": [], "image_loss": []}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动数据到设备
                input_text = batch['input_text']
                input_image = batch['input_image'].to(self.device)
                target_texts = batch['target_texts']
                target_images = [img.to(self.device) for img in batch['target_images']]
                
                # 前向传播
                loss_dict = self.model.forward_autoregressive_training(
                    input_text=input_text,
                    input_image=input_image,
                    target_texts=target_texts,
                    target_images=target_images,
                    tokenizer=self.tokenizer,
                    vae_model=self.vae_model,
                )
                
                total_losses["text_loss"].append(loss_dict["text_loss"].item())
                total_losses["image_loss"].append(loss_dict["image_loss"].item())
        
        # 计算平均损失
        avg_text_loss = sum(total_losses["text_loss"]) / len(total_losses["text_loss"])
        avg_image_loss = sum(total_losses["image_loss"]) / len(total_losses["image_loss"])
        
        self.logger.info(f"验证结果 - Text Loss: {avg_text_loss:.6f}, Image Loss: {avg_image_loss:.6f}")
        
        # 恢复训练模式
        self.model.train()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        self.logger.info(f"成功加载检查点，当前步数: {self.global_step}")
