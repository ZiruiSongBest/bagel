#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import traceback
from typing import Dict, Any, List
from data.dataset_base import PackedDataset, DataConfig

def debug_data_sample():
    """调试数据样本格式"""
    print("=== 调试数据样本格式 ===")
    
    # 简化版本，直接读取JSON数据
    try:
        with open('/remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json', 'r') as f:
            data = json.load(f)
        
        print(f"原始数据样本数: {len(data)}")
        print(f"第一个样本: {data[0]}")
        
        return data[0]
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        traceback.print_exc()
        return None

def debug_unified_trainer_data():
    """调试统一训练器的数据处理"""
    print("\n=== 调试统一训练器数据处理 ===")
    
    from training.unified_trainer import UnifiedTrainer
    from training.unified_data_processor import UnifiedGenerationDataset, create_unified_dataloader
    from modeling.bagel import BagelConfig, Bagel
    
    try:
        # 创建虚拟模型配置
        config = BagelConfig()
        
        # 创建数据集
        train_dataset = UnifiedGenerationDataset(
            data_path='/remote-home/hanmf/test/bagel/dataset/demo/demo_sample/anno.json',
            max_length=8192,
            data_format='conversation'
        )
        
        print(f"数据集大小: {len(train_dataset)}")
        
        # 创建数据加载器
        train_dataloader = create_unified_dataloader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # 获取一个batch
        for batch in train_dataloader:
            print(f"\nbatch键: {list(batch.keys())}")
            
            # 检查损失掩码
            if 'text_loss_mask' in batch:
                text_mask = batch['text_loss_mask']
                print(f"text_loss_mask 形状: {text_mask.shape}, 类型: {text_mask.dtype}")
                print(f"text_loss_mask 有效位置数: {text_mask.sum().item()}")
                
            if 'image_loss_mask' in batch:
                image_mask = batch['image_loss_mask']
                print(f"image_loss_mask 形状: {image_mask.shape}, 类型: {image_mask.dtype}")
                print(f"image_loss_mask 有效位置数: {image_mask.sum().item()}")
                
            # 检查packed_label_ids
            if 'packed_label_ids' in batch:
                labels = batch['packed_label_ids']
                print(f"packed_label_ids 形状: {labels.shape}, 类型: {labels.dtype}")
                
            break
            
    except Exception as e:
        print(f"❌ 统一训练器数据调试失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 调试原始数据集
    sample = debug_data_sample()
    
    # 调试统一训练器数据
    debug_unified_trainer_data()
