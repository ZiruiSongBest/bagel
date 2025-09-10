#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成训练的数据处理器

这个模块负责处理包含文本和图像的混合序列数据，将其转换为模型可以训练的格式。
支持的数据格式包括：
1. 文本 -> 图像 (Text-to-Image)
2. 图像 -> 文本 (Image-to-Text) 
3. 文本 -> 文本 -> 图像 (Multi-turn generation)
4. 图像 -> 文本 -> 图像 (Image editing with description)
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
class UnifiedTrainingExample:
    """统一生成训练样本的数据结构"""
    # 输入序列：可以是文本字符串或PIL图像的混合列表
    input_sequence: List[Union[str, Image.Image]]
    # 目标序列：期望生成的文本和图像混合列表
    target_sequence: List[Union[str, Image.Image]]
    # 序列中每个元素的类型标记 ('text' 或 'image')
    input_types: List[str] 
    target_types: List[str]
    # 可选的元数据
    metadata: Optional[Dict[str, Any]] = None


class UnifiedGenerationDataset(Dataset):
    """统一生成训练数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        vae_transform: ImageTransform,
        vit_transform: ImageTransform,
        new_token_ids: Dict[str, int],
        max_sequence_length: int = 2048,
        max_image_tokens: int = 1024,
        image_generation_prob: float = 0.5,
        text_generation_prob: float = 0.5,
        multimodal_prob: float = 0.3,
    ):
        """
        Args:
            data_path: 数据文件路径或目录
            tokenizer: 文本分词器
            vae_transform: VAE图像变换
            vit_transform: VIT图像变换 
            new_token_ids: 特殊token的ID映射
            max_sequence_length: 最大序列长度
            max_image_tokens: 单个图像的最大token数
            image_generation_prob: 图像生成任务的概率
            text_generation_prob: 文本生成任务的概率
            multimodal_prob: 多模态混合任务的概率
        """
        self.data_path = data_path  # 保存数据路径用于相对路径解析
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.max_sequence_length = max_sequence_length
        self.max_image_tokens = max_image_tokens
        self.image_generation_prob = image_generation_prob
        self.text_generation_prob = text_generation_prob
        self.multimodal_prob = multimodal_prob
        
        # 加载训练数据
        self.examples = self._load_data(data_path)
        print(f"加载了 {len(self.examples)} 个训练样本")
    
    def _load_data(self, data_path: str) -> List[UnifiedTrainingExample]:
        """加载训练数据"""
        examples = []
        
        if os.path.isfile(data_path):
            # 单个文件
            examples.extend(self._load_from_file(data_path))
        elif os.path.isdir(data_path):
            # 目录中的多个文件
            for filename in os.listdir(data_path):
                if filename.endswith(('.json', '.jsonl')):
                    filepath = os.path.join(data_path, filename)
                    examples.extend(self._load_from_file(filepath))
        else:
            raise ValueError(f"数据路径不存在: {data_path}")
        
        return examples
    
    def _load_from_file(self, filepath: str) -> List[UnifiedTrainingExample]:
        """从单个文件加载数据"""
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
            else:  # .jsonl
                data = [json.loads(line.strip()) for line in f if line.strip()]
        
        for item in data:
            try:
                example = self._parse_data_item(item)
                if example:
                    examples.append(example)
            except Exception as e:
                print(f"解析数据项时出错: {e}")
                continue
        
        return examples
    
    def _parse_data_item(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析单个数据项"""
        try:
            # 支持多种数据格式
            if 'messages' in item and 'images' in item:
                # 你的训练数据格式：包含messages和images
                return self._parse_messages_images_format(item)
            elif 'conversations' in item:
                # 对话格式
                return self._parse_conversation_format(item)
            elif 'input_sequence' in item and 'target_sequence' in item:
                # 直接格式
                return self._parse_direct_format(item)
            elif 'image_path' in item and 'caption' in item:
                # 图像描述格式
                return self._parse_image_caption_format(item)
            elif 'text_prompt' in item and 'image_path' in item:
                # 文本到图像格式
                return self._parse_text2image_format(item)
            else:
                print(f"未知的数据格式: {list(item.keys())}")
                return None
        except Exception as e:
            print(f"解析数据项失败: {e}")
            return None
    
    def _parse_messages_images_format(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析你的训练数据格式：messages + images"""
        try:
            messages = item['messages']
            images = item['images']
            
            # 将图像路径转换为Image对象
            image_objects = []
            for img_path in images:
                # 尝试相对于数据文件的路径
                if not os.path.exists(img_path):
                    # 尝试相对于数据文件目录的路径
                    data_dir = os.path.dirname(self.data_path) if hasattr(self, 'data_path') else ''
                    if data_dir:
                        alt_img_path = os.path.join(data_dir, img_path)
                        if os.path.exists(alt_img_path):
                            img_path = alt_img_path
                
                if os.path.exists(img_path):
                    image_objects.append(Image.open(img_path).convert('RGB'))
                else:
                    print(f"警告: 图像文件不存在: {img_path}")
                    # 为测试创建一个占位图像
                    placeholder_img = Image.new('RGB', (256, 256), color=(128, 128, 128))
                    image_objects.append(placeholder_img)
                    print(f"  使用占位图像替代")
            
            # 解析对话内容
            input_sequence = []
            target_sequence = []
            input_types = []
            target_types = []
            
            # 修复图像映射逻辑：
            # images[0] (first_image) 对应用户输入的text
            # images[1:] 对应助手回答中的<image>标记
            
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                
                if role == 'user':
                    # 用户输入包含文本和对应的第一张图像（作为上下文）
                    input_sequence.append(content)
                    input_types.append('text')
                    
                    # 添加对应的first_image作为输入上下文
                    if len(image_objects) > 0:
                        input_sequence.append(image_objects[0])  # first_frame作为输入
                        input_types.append('image')
                    
                elif role == 'assistant':
                    # 助手输出，可能包含文本和<image>标记
                    # <image>标记对应images[1:]（middle_frame, last_frame等）
                    assistant_images = image_objects[1:] if len(image_objects) > 1 else []
                    
                    parsed_content = self._parse_content_with_image_tags(
                        content, assistant_images, 0  # 从0开始计数assistant的图像
                    )
                    
                    for item_content, item_type in parsed_content:
                        target_sequence.append(item_content)
                        target_types.append(item_type)
            
            if input_sequence and target_sequence:
                # 添加调试信息验证图像映射
                print(f"数据解析结果:")
                print(f"  输入序列长度: {len(input_sequence)}, 类型: {input_types}")
                print(f"  目标序列长度: {len(target_sequence)}, 类型: {target_types}")
                print(f"  图像文件: {[os.path.basename(img) for img in images]}")
                
                return UnifiedTrainingExample(
                    input_sequence=input_sequence,
                    target_sequence=target_sequence,
                    input_types=input_types,
                    target_types=target_types,
                    metadata={}
                )
            
        except Exception as e:
            print(f"解析messages+images格式失败: {e}")
            
        return None
    
    def _parse_content_with_image_tags(self, content: str, image_objects: List[Image.Image], start_counter: int):
        """解析包含<image>标记的内容"""
        import re
        
        # 分割文本，找到<image>标记的位置
        parts = re.split(r'(<image>)', content)
        
        result = []
        image_counter = start_counter
        
        for part in parts:
            if part == '<image>':
                # 这是一个图像标记，替换为实际图像
                if image_counter < len(image_objects):
                    result.append((image_objects[image_counter], 'image'))
                    image_counter += 1
                else:
                    print(f"警告: 图像标记超出可用图像数量")
            elif part.strip():
                # 这是文本内容
                result.append((part.strip(), 'text'))
        
        return result
    
    def _parse_conversation_format(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析对话格式的数据"""
        conversations = item['conversations']
        input_sequence = []
        target_sequence = []
        input_types = []
        target_types = []
        
        for i, conv in enumerate(conversations):
            content = conv.get('content', '')
            conv_type = conv.get('type', 'text')
            
            if conv.get('role') == 'user':
                if conv_type == 'image' and 'image_path' in conv:
                    # 用户输入图像
                    img_path = conv['image_path']
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                        input_sequence.append(image)
                        input_types.append('image')
                elif conv_type == 'text':
                    # 用户输入文本
                    input_sequence.append(content)
                    input_types.append('text')
                    
            elif conv.get('role') == 'assistant':
                if conv_type == 'image' and 'image_path' in conv:
                    # 助手生成图像
                    img_path = conv['image_path']
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                        target_sequence.append(image)
                        target_types.append('image')
                elif conv_type == 'text':
                    # 助手生成文本
                    target_sequence.append(content)
                    target_types.append('text')
        
        if input_sequence and target_sequence:
            return UnifiedTrainingExample(
                input_sequence=input_sequence,
                target_sequence=target_sequence,
                input_types=input_types,
                target_types=target_types,
                metadata=item.get('metadata', {})
            )
        return None
    
    def _parse_direct_format(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析直接格式的数据"""
        input_seq = []
        target_seq = []
        input_types = []
        target_types = []
        
        for inp in item['input_sequence']:
            if inp['type'] == 'text':
                input_seq.append(inp['content'])
                input_types.append('text')
            elif inp['type'] == 'image':
                img_path = inp['image_path']
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    input_seq.append(image)
                    input_types.append('image')
        
        for tgt in item['target_sequence']:
            if tgt['type'] == 'text':
                target_seq.append(tgt['content'])
                target_types.append('text')
            elif tgt['type'] == 'image':
                img_path = tgt['image_path']
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    target_seq.append(image)
                    target_types.append('image')
        
        if input_seq and target_seq:
            return UnifiedTrainingExample(
                input_sequence=input_seq,
                target_sequence=target_seq,
                input_types=input_types,
                target_types=target_types,
                metadata=item.get('metadata', {})
            )
        return None
    
    def _parse_image_caption_format(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析图像描述格式的数据 (图像 -> 文本)"""
        img_path = item['image_path']
        if not os.path.exists(img_path):
            return None
            
        image = Image.open(img_path).convert('RGB')
        caption = item['caption']
        
        return UnifiedTrainingExample(
            input_sequence=[image],
            target_sequence=[caption],
            input_types=['image'],
            target_types=['text'],
            metadata=item.get('metadata', {})
        )
    
    def _parse_text2image_format(self, item: Dict) -> Optional[UnifiedTrainingExample]:
        """解析文本到图像格式的数据 (文本 -> 图像)"""
        img_path = item['image_path']
        if not os.path.exists(img_path):
            return None
            
        image = Image.open(img_path).convert('RGB')
        text_prompt = item['text_prompt']
        
        return UnifiedTrainingExample(
            input_sequence=[text_prompt],
            target_sequence=[image],
            input_types=['text'],
            target_types=['image'],
            metadata=item.get('metadata', {})
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取训练样本"""
        example = self.examples[idx]
        
        # 转换为模型训练格式
        return self._convert_to_training_format(example)
    
    def _convert_to_training_format(self, example: UnifiedTrainingExample) -> Dict[str, Any]:
        """将训练样本转换为模型可处理的格式"""
        
        # 构建完整序列：输入 + 目标
        full_sequence = example.input_sequence + example.target_sequence
        full_types = example.input_types + example.target_types
        
        # 处理序列中的每个元素
        processed_data = self._process_mixed_sequence(full_sequence, full_types)
        
        # 创建损失掩码：只在目标序列上计算损失
        input_length = len(example.input_sequence)
        target_length = len(example.target_sequence)
        
        # 文本损失掩码
        text_loss_mask = torch.zeros(processed_data['sequence_length'], dtype=torch.bool)
        # 图像损失掩码  
        image_loss_mask = torch.zeros(processed_data['sequence_length'], dtype=torch.bool)
        
        # 标记目标序列位置进行损失计算
        self._create_loss_masks(
            processed_data, 
            example.target_types,
            input_length,
            text_loss_mask,
            image_loss_mask
        )
        
        return {
            **processed_data,
            'text_loss_mask': text_loss_mask,
            'image_loss_mask': image_loss_mask,
            'input_length': input_length,
            'target_length': target_length,
            'metadata': example.metadata or {}
        }
    
    def _process_mixed_sequence(self, sequence: List, types: List[str]) -> Dict[str, Any]:
        """处理混合序列（文本+图像）"""
        
        # 初始化各种列表来存储处理结果
        packed_text_ids = []
        packed_text_indexes = []
        packed_vit_tokens = []
        packed_vit_token_indexes = []
        packed_vit_position_ids = []
        vit_token_seqlens = []
        packed_vae_images = []
        packed_vae_token_indexes = []
        packed_vae_position_ids = []
        patchified_vae_latent_shapes = []
        packed_position_ids = []
        
        current_index = 0
        current_position = 0
        
        for item, item_type in zip(sequence, types):
            if item_type == 'text':
                # 处理文本
                text_data = self._process_text_item(
                    item, current_index, current_position
                )
                packed_text_ids.extend(text_data['text_ids'])
                packed_text_indexes.extend(text_data['text_indexes'])
                packed_position_ids.extend(text_data['position_ids'])
                
                current_index += len(text_data['text_ids'])
                current_position += 1
                
            elif item_type == 'image':
                # 处理图像 - 同时准备VIT和VAE数据
                image_data = self._process_image_item(
                    item, current_index, current_position
                )
                
                # VIT数据（用于理解）
                vit_num_tokens = 0
                if image_data['vit_tokens'] is not None:
                    vit_start_idx = current_index + 1  # +1 for start_of_image token
                    packed_vit_tokens.append(image_data['vit_tokens'])
                    vit_num_tokens = image_data['vit_tokens'].shape[0]
                    packed_vit_token_indexes.extend(
                        range(vit_start_idx, vit_start_idx + vit_num_tokens)
                    )
                    packed_vit_position_ids.append(image_data['vit_position_ids'])
                
                # 始终添加到 vit_token_seqlens，即使没有VIT数据也添加0
                vit_token_seqlens.append(vit_num_tokens)
                
                # VAE数据（用于生成）
                vae_num_tokens = 0
                if image_data['vae_image'] is not None:
                    vae_start_idx = current_index + 1  # +1 for start_of_image token
                    packed_vae_images.append(image_data['vae_image'])
                    vae_num_tokens = image_data['vae_num_tokens']
                    packed_vae_token_indexes.extend(
                        range(vae_start_idx + vit_num_tokens, vae_start_idx + vit_num_tokens + vae_num_tokens)
                    )
                    packed_vae_position_ids.append(image_data['vae_position_ids'])
                    patchified_vae_latent_shapes.append(image_data['vae_latent_shape'])
                
                # 文本tokens (start_of_image + end_of_image)
                # 修正索引：start_of_image在开始，end_of_image在所有图像tokens之后
                start_of_image_idx = current_index
                end_of_image_idx = current_index + 1 + vit_num_tokens + vae_num_tokens
                
                packed_text_ids.extend(image_data['text_ids'])
                packed_text_indexes.extend([start_of_image_idx, end_of_image_idx])
                packed_position_ids.extend([current_position] * (2 + vit_num_tokens + vae_num_tokens))
                
                current_index += image_data['total_tokens']
                current_position += 1
        
        # 创建简单的因果注意力掩码
        from data.data_utils import prepare_attention_mask_per_sample
        nested_attention_mask = prepare_attention_mask_per_sample(
            [current_index], ['causal'], 'cpu'
        )
        
        # 构建最终数据
        result = {
            'sequence_length': current_index,
            'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long),
            'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long),
            'packed_position_ids': torch.tensor(packed_position_ids, dtype=torch.long),
            # 添加注意力掩码相关参数
            'sample_lens': [current_index],  # 当前样本的总长度
            'split_lens': [current_index],   # 将整个序列作为一个split  
            'attn_modes': ['causal'],        # 使用因果注意力
            'nested_attention_masks': [nested_attention_mask],  # 直接提供注意力掩码
        }
        
        # 添加VIT数据（如果有）
        if packed_vit_tokens:
            result.update({
                'packed_vit_tokens': torch.cat(packed_vit_tokens, dim=0),
                'packed_vit_token_indexes': torch.tensor(packed_vit_token_indexes, dtype=torch.long),
                'packed_vit_position_ids': torch.cat(packed_vit_position_ids, dim=0),
                'vit_token_seqlens': torch.tensor(vit_token_seqlens, dtype=torch.int),
            })
        
        # 添加VAE数据（如果有）
        if packed_vae_images:
            # 对图像进行padding以适应batch处理
            image_sizes = [img.shape for img in packed_vae_images]
            max_image_size = [max(sizes) for sizes in zip(*image_sizes)]
            padded_images = torch.zeros(len(packed_vae_images), *max_image_size)
            
            for i, img in enumerate(packed_vae_images):
                padded_images[i, :, :img.shape[1], :img.shape[2]] = img
                
            result.update({
                'padded_vae_images': padded_images,
                'packed_vae_token_indexes': torch.tensor(packed_vae_token_indexes, dtype=torch.long),
                'packed_vae_position_ids': torch.cat(packed_vae_position_ids, dim=0),
                'patchified_vae_latent_shapes': patchified_vae_latent_shapes,
                'packed_timesteps': torch.tensor([0.0] * len(packed_vae_token_indexes)),  # 训练时使用clean images
            })
        
        return result
    
    def _process_text_item(self, text: str, start_index: int, position: int) -> Dict[str, Any]:
        """处理单个文本项"""
        # 对文本进行tokenization
        text_ids = self.tokenizer.encode(text)
        
        # 添加特殊tokens
        text_ids = [self.new_token_ids['bos_token_id']] + text_ids + [self.new_token_ids['eos_token_id']]
        
        # 生成对应的索引和位置
        text_indexes = list(range(start_index, start_index + len(text_ids)))
        position_ids = [position] * len(text_ids)
        
        return {
            'text_ids': text_ids,
            'text_indexes': text_indexes,
            'position_ids': position_ids,
        }
    
    def _process_image_item(self, image: Image.Image, start_index: int, position: int) -> Dict[str, Any]:
        """处理单个图像项"""
        
        # VIT处理（用于理解）
        vit_image = self.vit_transform(image)
        from data.data_utils import patchify, get_flattened_position_ids_extrapolate
        
        vit_tokens = patchify(vit_image, patch_size=14)  # VIT patch size
        vit_position_ids = get_flattened_position_ids_extrapolate(
            vit_image.size(1), vit_image.size(2), 
            patch_size=14, 
            max_num_patches_per_side=70
        )
        
        # VAE处理（用于生成）
        vae_image = self.vae_transform(image)
        
        # 计算VAE latent的形状
        H, W = vae_image.shape[1:]
        latent_downsample = 32  # VAE downsample ratio * patch size  
        h = H // latent_downsample
        w = W // latent_downsample
        vae_num_tokens = h * w
        
        vae_position_ids = get_flattened_position_ids_extrapolate(
            H, W,
            latent_downsample,
            max_num_patches_per_side=64
        )
        
        # 图像的文本tokens: start_of_image + end_of_image (对应 <|vision_start|> 和 <|vision_end|>)
        image_text_ids = [
            self.new_token_ids['start_of_image'],  # <|vision_start|>
            self.new_token_ids['end_of_image']     # <|vision_end|>
        ]
        
        # 总token数: start_of_image + vit_tokens + vae_tokens + end_of_image
        total_tokens = 1 + vit_tokens.shape[0] + vae_num_tokens + 1
        
        # 文本token的索引
        text_indexes = [start_index, start_index + total_tokens - 1]
        position_ids = [position] * total_tokens
        
        return {
            'vit_tokens': vit_tokens,
            'vit_position_ids': vit_position_ids,
            'vae_image': vae_image,
            'vae_num_tokens': vae_num_tokens,
            'vae_position_ids': vae_position_ids,
            'vae_latent_shape': (h, w),
            'text_ids': image_text_ids,
            'text_indexes': text_indexes,
            'position_ids': position_ids,
            'total_tokens': total_tokens,
        }
    
    def _create_loss_masks(
        self, 
        processed_data: Dict[str, Any], 
        target_types: List[str],
        input_length: int,
        text_loss_mask: torch.Tensor,
        image_loss_mask: torch.Tensor
    ):
        """创建损失计算的掩码"""
        
        # 简化版本：假设目标序列在输入序列之后
        # 在实际实现中，你可能需要更精确地跟踪每个元素的位置
        
        sequence_length = processed_data['sequence_length']
        
        # 计算目标序列的大概位置（这里需要根据具体的序列结构调整）
        target_start_idx = sequence_length // 2  # 简化假设
        
        # 标记文本和图像的损失位置
        for i, target_type in enumerate(target_types):
            if target_type == 'text':
                # 标记文本位置
                text_loss_mask[target_start_idx:] = True
            elif target_type == 'image':
                # 标记图像位置  
                if 'packed_vae_token_indexes' in processed_data:
                    vae_indexes = processed_data['packed_vae_token_indexes']
                    image_loss_mask[vae_indexes] = True


def create_unified_dataloader(
    dataset: UnifiedGenerationDataset,
    batch_size: int = 1,  # 建议batch_size=1，因为序列长度差异很大
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn=None,
    sampler=None,
) -> DataLoader:
    """创建统一生成的数据加载器"""
    
    if collate_fn is None:
        collate_fn = unified_collate_fn
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def unified_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统一生成的批次整理函数"""
    
    if len(batch) == 1:
        # 单样本直接返回
        sample = batch[0]
        
        # 确保必要的键存在，用于注意力掩码生成
        required_keys = ['sample_lens', 'split_lens', 'attn_modes']
        for key in required_keys:
            if key not in sample:
                if key == 'sample_lens':
                    sample[key] = [sample.get('sequence_length', 0)]
                elif key == 'split_lens':
                    sample[key] = [sample.get('sequence_length', 0)]
                elif key == 'attn_modes':
                    sample[key] = ['causal']
        
        return sample
    
    # 多样本的batch处理（如果需要）
    # 由于序列长度差异很大，建议使用batch_size=1
    raise NotImplementedError("多样本batch处理尚未实现，建议使用batch_size=1")


# 数据格式示例
EXAMPLE_DATA_FORMATS = {
    "对话格式": {
        "conversations": [
            {
                "role": "user",
                "type": "text", 
                "content": "请生成一张猫的图片"
            },
            {
                "role": "assistant",
                "type": "image",
                "image_path": "/path/to/cat_image.jpg"
            }
        ]
    },
    
    "直接格式": {
        "input_sequence": [
            {"type": "text", "content": "Edit this image: add a hat to the cat"}
        ],
        "target_sequence": [
            {"type": "image", "image_path": "/path/to/cat_with_hat.jpg"}
        ]
    },
    
    "图像描述格式": {
        "image_path": "/path/to/image.jpg",
        "caption": "A beautiful sunset over the ocean"
    },
    
    "文本到图像格式": {
        "text_prompt": "A red car driving down the highway",
        "image_path": "/path/to/car_image.jpg"
    }
}


if __name__ == "__main__":
    # 测试数据处理器
    print("统一生成数据处理器测试")
    print("支持的数据格式:")
    for format_name, example in EXAMPLE_DATA_FORMATS.items():
        print(f"\n{format_name}:")
        print(json.dumps(example, indent=2, ensure_ascii=False))
