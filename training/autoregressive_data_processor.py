#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自回归序列生成的数据处理器

支持多模态自回归训练，其中模型需要逐步生成：
文本 -> 图像 -> 文本 -> 图像 的序列
"""

import os
import json
import torch
import random
from typing import List, Dict, Union, Optional, Tuple, Any
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from data.data_utils import pil_img2rgb
from data.transforms import ImageTransform


@dataclass
class AutoregressiveTrainingExample:
    """自回归训练样本的数据结构"""
    # 输入文本和图像
    input_text: str
    input_image: Image.Image
    
    # 目标序列：交替的文本和图像
    target_texts: List[str]  # ["第一段思考", "第二段思考"]
    target_images: List[Image.Image]  # [中间图像, 最终图像]
    
    # 可选的元数据
    metadata: Optional[Dict[str, Any]] = None


class AutoregressiveDataset(Dataset):
    """自回归序列生成数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        vae_transform: ImageTransform,
        vit_transform: ImageTransform,
        new_token_ids: Dict[str, int],
        max_sequence_length: int = 2048,
    ):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: 文本分词器
            vae_transform: VAE图像变换（用于目标图像）
            vit_transform: VIT图像变换（用于输入图像）
            new_token_ids: 特殊token的ID映射
            max_sequence_length: 最大序列长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.max_sequence_length = max_sequence_length
        
        # 加载训练数据
        self.examples = self._load_data(data_path)
        print(f"加载了 {len(self.examples)} 个自回归训练样本")
    
    def _load_data(self, data_path: str) -> List[AutoregressiveTrainingExample]:
        """加载训练数据"""
        examples = []
        
        if os.path.isfile(data_path):
            examples.extend(self._load_from_file(data_path))
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                if filename.endswith(('.json', '.jsonl')):
                    filepath = os.path.join(data_path, filename)
                    examples.extend(self._load_from_file(filepath))
        else:
            raise ValueError(f"数据路径不存在: {data_path}")
        
        return examples
    
    def _load_from_file(self, filepath: str) -> List[AutoregressiveTrainingExample]:
        """从文件加载数据"""
        examples = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.jsonl'):
                    # JSONL格式
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            example = self._parse_item(item)
                            if example:
                                examples.append(example)
                else:
                    # JSON格式
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            example = self._parse_item(item)
                            if example:
                                examples.append(example)
                    else:
                        example = self._parse_item(data)
                        if example:
                            examples.append(example)
        except Exception as e:
            print(f"加载文件 {filepath} 时出错: {e}")
        
        return examples
    
    def _parse_item(self, item: Dict) -> Optional[AutoregressiveTrainingExample]:
        """解析单个数据项"""
        try:
            # 解析messages+images格式
            if 'messages' in item and 'images' in item:
                return self._parse_messages_format(item)
            
            # 解析其他格式...
            # 可以在这里添加更多格式的解析
            
        except Exception as e:
            print(f"解析数据项失败: {e}")
        
        return None
    
    def _parse_messages_format(self, item: Dict) -> Optional[AutoregressiveTrainingExample]:
        """解析messages+images格式的数据"""
        try:
            messages = item['messages']
            image_paths = item['images']
            
            # 加载所有图像
            image_objects = []
            base_dir = os.path.dirname(self.data_path) if os.path.isfile(self.data_path) else self.data_path
            
            for img_path in image_paths:
                if not os.path.isabs(img_path):
                    full_img_path = os.path.join(base_dir, img_path)
                else:
                    full_img_path = img_path
                
                if os.path.exists(full_img_path):
                    try:
                        image = Image.open(full_img_path).convert('RGB')
                        image_objects.append(image)
                    except Exception as e:
                        print(f"无法加载图像 {full_img_path}: {e}")
                        continue
            
            if len(image_objects) < 3:  # 需要至少3张图：输入图 + 中间图 + 最终图
                print(f"图像数量不足，需要至少3张图像，实际: {len(image_objects)}")
                return None
            
            # 解析对话内容
            input_text = ""
            assistant_content = ""
            
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                
                if role == 'user':
                    input_text = content
                elif role == 'assistant':
                    assistant_content = content
            
            if not input_text or not assistant_content:
                print("缺少用户输入或助手回复")
                return None
            
            # 解析助手回复中的文本和图像
            target_texts, target_images = self._parse_assistant_content(
                assistant_content, image_objects[1:]  # 跳过第一张图（输入图）
            )
            
            if not target_texts or not target_images:
                print("无法解析目标文本或图像")
                return None
            
            return AutoregressiveTrainingExample(
                input_text=input_text,
                input_image=image_objects[0],  # 第一张图作为输入
                target_texts=target_texts,
                target_images=target_images,
                metadata=item.get('metadata', {})
            )
            
        except Exception as e:
            print(f"解析messages+images格式失败: {e}")
            
        return None
    
    def _parse_assistant_content(self, content: str, images: List[Image.Image]) -> Tuple[List[str], List[Image.Image]]:
        """解析助手回复内容，提取文本和图像"""
        import re
        
        # 使用正则表达式分割内容
        parts = re.split(r'<image>', content)
        
        texts = []
        images_used = []
        
        # 第一部分总是文本
        if parts[0].strip():
            texts.append(parts[0].strip())
        
        # 处理后续部分
        image_counter = 0
        for i in range(1, len(parts)):
            # 每个<image>标记对应一张图像
            if image_counter < len(images):
                images_used.append(images[image_counter])
                image_counter += 1
            
            # <image>标记后的文本
            if parts[i].strip():
                texts.append(parts[i].strip())
        
        return texts, images_used
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取训练样本"""
        example = self.examples[idx]
        
        # 处理输入图像（使用VIT transform）
        input_image = self.vit_transform(example.input_image)
        
        # 处理目标图像（使用VAE transform）
        target_images = []
        for img in example.target_images:
            target_images.append(self.vae_transform(img))
        
        return {
            'input_text': example.input_text,
            'input_image': input_image,
            'target_texts': example.target_texts,
            'target_images': target_images,
            'metadata': example.metadata or {}
        }


def collate_autoregressive_batch(batch):
    """
    自回归批处理函数
    
    注意：由于自回归的复杂性，暂时只支持batch_size=1
    """
    if len(batch) != 1:
        raise ValueError("自回归训练当前只支持batch_size=1")
    
    return batch[0]


# 测试代码
if __name__ == "__main__":
    # 这里可以添加一些测试代码
    pass
