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
import logging

from data.data_utils import pil_img2rgb
from data.transforms import ImageTransform

# 设置logger
logger = logging.getLogger(__name__)

# 默认统一系统提示词，与推理阶段保持一致
DEFAULT_SYSTEM_PROMPT = '''You are a multimodal AI that can generate both text and images. When generating content:

1. For text: Simply generate text naturally
2. For images: Use the special tokens <|vision_start|> and <|vision_end|> to mark where an image should be generated

- Mixed response: "I'll create an image for you: <|vision_start|> <|vision_end|> And here's some additional text."

Always use <|vision_start|> and <|vision_end|> tokens when you want to generate an image.'''


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
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
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
        self.system_prompt = system_prompt
        
        # 加载训练数据
        self.examples = self._load_data(data_path)
        logger.info(f"加载了 {len(self.examples)} 个训练样本")
        # print(f"加载了 {len(self.examples)} 个训练样本")  # 注释掉控制台输出

        # 只提示一次：当目标图像之前缺乏可监督token时，用于调试<|vision_start|>
        self._warned_start_without_context = False
    
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
                logger.warning(f"解析数据项时出错: {e}")
                # print(f"解析数据项时出错: {e}")  # 注释掉控制台输出
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
                logger.warning(f"未知的数据格式: {list(item.keys())}")
                # print(f"未知的数据格式: {list(item.keys())}")  # 注释掉控制台输出
                return None
        except Exception as e:
            logger.error(f"解析数据项失败: {e}")
            # print(f"解析数据项失败: {e}")  # 注释掉控制台输出
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
                    logger.warning(f"图像文件不存在: {img_path}")
                    # print(f"警告: 图像文件不存在: {img_path}")  # 注释掉控制台输出
                    # 为测试创建一个占位图像
                    placeholder_img = Image.new('RGB', (256, 256), color=(128, 128, 128))
                    image_objects.append(placeholder_img)
                    logger.info(f"使用占位图像替代: {img_path}")
                    # print(f"  使用占位图像替代")  # 注释掉控制台输出
            
            # 解析对话内容
            input_sequence = []
            target_sequence = []
            input_types = []
            target_types = []

            if self.system_prompt:
                input_sequence.append(self.system_prompt)
                input_types.append('text')
            
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
                        content,
                        assistant_images,
                        0,  # 从0开始计数assistant的图像
                        wrap_with_role=True,
                    )
                    
                    for item_content, item_type in parsed_content:
                        target_sequence.append(item_content)
                        target_types.append(item_type)
            
            if input_sequence and target_sequence:
                # 将调试信息写入logger而不是控制台
                logger.debug(f"数据解析结果:")
                logger.debug(f"  输入序列长度: {len(input_sequence)}, 类型: {input_types}")
                logger.debug(f"  目标序列长度: {len(target_sequence)}, 类型: {target_types}")
                logger.debug(f"  图像文件: {[os.path.basename(img) for img in images]}")
                # print(f"数据解析结果:")  # 注释掉控制台输出
                # print(f"  输入序列长度: {len(input_sequence)}, 类型: {input_types}")
                # print(f"  目标序列长度: {len(target_sequence)}, 类型: {target_types}")
                # print(f"  图像文件: {[os.path.basename(img) for img in images]}")
                
                return UnifiedTrainingExample(
                    input_sequence=input_sequence,
                    target_sequence=target_sequence,
                    input_types=input_types,
                    target_types=target_types,
                    metadata={}
                )
            
        except Exception as e:
            logger.error(f"解析messages+images格式失败: {e}")
            # print(f"解析messages+images格式失败: {e}")  # 注释掉控制台输出
            
        return None
    
    def _parse_content_with_image_tags(
        self,
        content: str,
        image_objects: List[Image.Image],
        start_counter: int,
        wrap_with_role: bool = False,
    ):
        """解析包含<image>标记的内容"""
        import re

        # 分割文本，找到<image>标记的位置
        parts = re.split(r'(<image>)', content)

        result = []
        image_counter = start_counter

        text_indices = [
            idx for idx, part in enumerate(parts)
            if part != '<image>' and part.strip()
        ]
        first_text_idx = text_indices[0] if text_indices else None
        last_text_idx = text_indices[-1] if text_indices else None

        for idx, part in enumerate(parts):
            if part == '<image>':
                # 这是一个图像标记，替换为实际图像
                if image_counter < len(image_objects):
                    result.append((image_objects[image_counter], 'image'))
                    image_counter += 1
                else:
                    logger.warning(f"图像标记超出可用图像数量")
                    # print(f"警告: 图像标记超出可用图像数量")  # 注释掉控制台输出
            elif part.strip():
                # 这是文本内容
                stripped = part.strip()
                if wrap_with_role and first_text_idx is not None:
                    if idx == first_text_idx:
                        stripped = 'role": "assistant", "content": "{}'.format(stripped)
                    if idx == last_text_idx:
                        stripped = f'{stripped}"'
                result.append((stripped, 'text'))

        if wrap_with_role and not text_indices:
            result.insert(0, ('role": "assistant", "content": ""', 'text'))

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
        target_flags = [False] * len(example.input_sequence) + [True] * len(example.target_sequence)

        # 处理序列中的每个元素
        processed_data = self._process_mixed_sequence(full_sequence, full_types, target_flags)

        input_length = len(example.input_sequence)
        target_length = len(example.target_sequence)

        text_loss_mask = processed_data.pop('text_loss_mask')
        image_loss_mask = processed_data.pop('image_loss_mask')

        return {
            **processed_data,
            'text_loss_mask': text_loss_mask,
            'image_loss_mask': image_loss_mask,
            'input_length': input_length,
            'target_length': target_length,
            'metadata': example.metadata or {}
        }
    
    def _process_mixed_sequence(
        self,
        sequence: List,
        types: List[str],
        target_flags: List[bool],
    ) -> Dict[str, Any]:
        """处理混合序列（文本+图像）"""
        
        # 初始化各种列表来存储处理结果
        packed_text_ids = []
        packed_text_indexes = []
        packed_label_ids = []
        packed_text_loss_mask = []
        packed_vit_tokens = []
        packed_vit_token_indexes = []
        packed_vit_position_ids = []
        vit_token_seqlens = []
        packed_vae_images = []
        packed_vae_token_indexes = []
        target_vae_token_indexes = []
        packed_vae_position_ids = []
        patchified_vae_latent_shapes = []
        packed_position_ids = []

        target_image_start_token_positions = []
        
        current_index = 0
        current_position = 0
        
        for item, item_type, is_target in zip(sequence, types, target_flags):
            if item_type == 'text':
                # 处理文本
                text_data = self._process_text_item(
                    item, current_index, current_position, is_target
                )
                packed_text_ids.extend(text_data['text_ids'])
                packed_text_indexes.extend(text_data['text_indexes'])
                packed_position_ids.extend(text_data['position_ids'])
                packed_label_ids.extend(text_data['label_ids'])
                packed_text_loss_mask.extend(text_data['loss_mask'])

                current_index += len(text_data['text_ids'])
                current_position += 1

            elif item_type == 'image':
                # 处理图像 - 同时准备VIT和VAE数据
                image_data = self._process_image_item(
                    item, current_index, current_position, is_target
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
                    vae_token_range = range(
                        vae_start_idx + vit_num_tokens,
                        vae_start_idx + vit_num_tokens + vae_num_tokens,
                    )
                    packed_vae_token_indexes.extend(vae_token_range)
                    packed_vae_position_ids.append(image_data['vae_position_ids'])
                    patchified_vae_latent_shapes.append(image_data['vae_latent_shape'])
                    if is_target:
                        target_vae_token_indexes.extend(vae_token_range)
                
                # 文本tokens (start_of_image + end_of_image)
                # 修正索引：start_of_image在开始，end_of_image在所有图像tokens之后
                start_of_image_idx = current_index
                end_of_image_idx = current_index + 1 + vit_num_tokens + vae_num_tokens

                start_token_list_idx = len(packed_text_ids)

                packed_text_ids.extend(image_data['text_ids'])
                packed_text_indexes.extend([start_of_image_idx, end_of_image_idx])
                packed_position_ids.extend([current_position] * (2 + vit_num_tokens + vae_num_tokens))
                packed_label_ids.extend(image_data['label_ids'])
                packed_text_loss_mask.extend(image_data['loss_mask'])

                if is_target:
                    target_image_start_token_positions.append(start_token_list_idx)

                current_index += image_data['total_tokens']
                current_position += 1
        
        # 创建简单的因果注意力掩码
        from data.data_utils import prepare_attention_mask_per_sample

        if target_image_start_token_positions:
            start_token_id = self.new_token_ids['start_of_image']
            disallowed_prev_ids = {
                self.new_token_ids['start_of_image'],
                self.new_token_ids['end_of_image'],
            }
            for start_position in target_image_start_token_positions:
                prev_idx = start_position - 1
                while prev_idx >= 0:
                    if not packed_text_loss_mask[prev_idx]:
                        prev_idx -= 1
                        continue
                    if packed_text_ids[prev_idx] in disallowed_prev_ids:
                        prev_idx -= 1
                        continue
                    break

                if prev_idx >= 0:
                    packed_label_ids[prev_idx] = start_token_id
                    packed_text_loss_mask[start_position] = False
                else:
                    if not self._warned_start_without_context:
                        logger.warning(
                            "目标图像前缺少可监督token，<|vision_start|>仍将在其自身位置上计算损失。"
                        )
                        self._warned_start_without_context = True

        nested_attention_mask = prepare_attention_mask_per_sample(
            [current_index], ['causal'], 'cpu'
        )

        # 文本/图像损失掩码（序列级别）
        text_loss_mask = torch.zeros(current_index, dtype=torch.bool)
        if packed_text_indexes:
            text_indexes_tensor = torch.tensor(packed_text_indexes, dtype=torch.long)
            loss_mask_tensor = torch.tensor(packed_text_loss_mask, dtype=torch.bool)
            if loss_mask_tensor.any():
                text_loss_mask[text_indexes_tensor[loss_mask_tensor]] = True

        image_loss_mask = torch.zeros(current_index, dtype=torch.bool)
        if target_vae_token_indexes:
            image_indexes_tensor = torch.tensor(target_vae_token_indexes, dtype=torch.long)
            image_loss_mask[image_indexes_tensor] = True

        # 构建packed_timesteps：仅对目标图像token保留非负timestep，其余设置为-Inf
        packed_timesteps = torch.tensor([], dtype=torch.float32)
        if packed_vae_token_indexes:
            target_index_set = set(target_vae_token_indexes)
            timestep_list = [0.0 if idx in target_index_set else float('-inf') for idx in packed_vae_token_indexes]
            packed_timesteps = torch.tensor(timestep_list, dtype=torch.float32)

        # 构建最终数据
        result = {
            'sequence_length': current_index,
            'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long),
            'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long),
            'packed_position_ids': torch.tensor(packed_position_ids, dtype=torch.long),
            'packed_label_ids': torch.tensor(packed_label_ids, dtype=torch.long),
            'packed_text_loss_mask': torch.tensor(packed_text_loss_mask, dtype=torch.bool),
            'text_loss_mask': text_loss_mask,
            'image_loss_mask': image_loss_mask,
            'mse_loss_indexes': image_loss_mask,
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
        elif vit_token_seqlens:
            # 即使没有实际的VIT tokens，如果有序列长度信息也要添加
            # 这种情况发生在所有图像的VIT tokens都是0时
            result.update({
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
                'packed_timesteps': packed_timesteps if packed_timesteps.numel() > 0 else torch.tensor([], dtype=torch.float32),
            })

        return result
    
    def _process_text_item(
        self,
        text: str,
        start_index: int,
        position: int,
        is_target: bool,
    ) -> Dict[str, Any]:
        """处理单个文本项并生成对应标签。"""
        # 关闭tokenizer自带的特殊tokens，统一由数据处理器控制
        text_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # 输入序列以BOS开头
        text_ids = [self.new_token_ids['bos_token_id']] + text_token_ids
        # 预测目标在时间维度上右移一个位置，并在末尾补EOS
        label_ids = text_token_ids + [self.new_token_ids['eos_token_id']]

        # 生成对应的索引和位置id
        text_indexes = list(range(start_index, start_index + len(text_ids)))
        position_ids = [position] * len(text_ids)

        # 是否对此片段计算文本CE损失
        loss_mask = [is_target] * len(text_ids)

        return {
            'text_ids': text_ids,
            'label_ids': label_ids,
            'text_indexes': text_indexes,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }
    
    def _process_image_item(
        self,
        image: Image.Image,
        start_index: int,
        position: int,
        is_target: bool,
    ) -> Dict[str, Any]:
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
            self.new_token_ids['end_of_image'],    # <|vision_end|>
        ]

        # 图像相关文本token的label会在外层进行重定向，使<|vision_start|>由前一个token预测
        image_label_ids = [
            self.new_token_ids['start_of_image'],
            self.new_token_ids['end_of_image'],
        ]
        image_loss_mask = [is_target, is_target]
        
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
            'label_ids': image_label_ids,
            'text_indexes': text_indexes,
            'position_ids': position_ids,
            'loss_mask': image_loss_mask,
            'total_tokens': total_tokens,
        }
    

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
