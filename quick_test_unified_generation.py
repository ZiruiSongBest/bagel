#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试统一多模态生成功能

这个脚本提供了一个简化的测试接口，用于快速验证统一生成功能
"""

import torch
from PIL import Image
import json
from typing import List, Union, Dict, Any

def quick_test_unified_generation(inferencer, test_prompt: str = None):
    """
    快速测试统一生成功能
    
    Args:
        inferencer: InterleaveInferencer实例
        test_prompt: 测试提示文本，如果为None则使用默认提示
    """
    if test_prompt is None:
        test_prompt = "请描述一个美丽的日出场景，然后画出这个场景。"
    
    print("=== 快速测试统一多模态生成 ===")
    print(f"测试提示: {test_prompt}")
    print("-" * 50)
    
    try:
        # 测试1: 获取原始token序列
        print("1. 获取原始token序列...")
        raw_tokens = inferencer.unified_generate(
            input_text=test_prompt,
            max_length=100,
            do_sample=False,
            temperature=1.0,
            return_raw_tokens=True
        )
        
        print(f"生成了 {len(raw_tokens)} 个token")
        print(f"前10个token: {raw_tokens[:10]}")
        
        # 检查是否包含图像token
        img_start_id = inferencer.new_token_ids.get('img_start_token_id')
        img_end_id = inferencer.new_token_ids.get('img_end_token_id')
        
        has_img_start = img_start_id in raw_tokens if img_start_id else False
        has_img_end = img_end_id in raw_tokens if img_end_id else False
        
        print(f"包含图像开始token: {has_img_start}")
        print(f"包含图像结束token: {has_img_end}")
        
    except Exception as e:
        print(f"原始token生成测试失败: {e}")
        return False
    
    try:
        # 测试2: 获取解析后的结果
        print("\n2. 获取解析后的结果...")
        parsed_result = inferencer.unified_generate(
            input_text=test_prompt,
            max_length=100,
            do_sample=False,
            temperature=1.0,
            return_raw_tokens=False
        )
        
        print(f"解析后共 {len(parsed_result)} 个元素")
        for i, item in enumerate(parsed_result):
            if isinstance(item, str):
                print(f"文本元素 {i}: {item[:50]}...")
            elif isinstance(item, Image.Image):
                print(f"图像元素 {i}: {item.size}")
            else:
                print(f"其他元素 {i}: {type(item)}")
        
    except Exception as e:
        print(f"解析结果测试失败: {e}")
        return False
    
    print("\n✅ 快速测试完成！")
    return True

def validate_token_ids(inferencer):
    """验证必要的token ID是否存在"""
    print("=== 验证Token ID配置 ===")
    
    required_tokens = [
        'img_start_token_id',
        'img_end_token_id', 
        'eos_token_id',
        'bos_token_id'
    ]
    
    missing_tokens = []
    for token_name in required_tokens:
        if token_name not in inferencer.new_token_ids:
            missing_tokens.append(token_name)
        else:
            token_id = inferencer.new_token_ids[token_name]
            print(f"✅ {token_name}: {token_id}")
    
    if missing_tokens:
        print(f"❌ 缺少必要的token ID: {missing_tokens}")
        return False
    else:
        print("✅ 所有必要的token ID都已配置")
        return True

def test_token_prediction(inferencer, test_text: str = "Hello"):
    """测试token预测功能"""
    print("=== 测试Token预测功能 ===")
    
    try:
        # 初始化上下文
        gen_context = inferencer.init_gen_context()
        gen_context = inferencer.update_context_text(test_text, gen_context)
        
        # 测试预测功能
        dummy_input_ids = torch.tensor([[1]], device=inferencer.model.device)
        kv_lens = torch.tensor(gen_context['kv_lens'], dtype=torch.int, device=inferencer.model.device)
        ropes = torch.tensor(gen_context['ropes'], dtype=torch.long, device=inferencer.model.device)
        
        logits, updated_kv = inferencer._predict_next_token_logits(
            input_ids=dummy_input_ids,
            past_key_values=gen_context['past_key_values'],
            kv_lens=kv_lens,
            ropes=ropes
        )
        
        print(f"✅ 预测logits形状: {logits.shape}")
        print(f"✅ 更新后的KV cache类型: {type(updated_kv)}")
        return True
        
    except Exception as e:
        print(f"❌ Token预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def comprehensive_test(inferencer):
    """综合测试"""
    print("开始综合测试统一多模态生成功能...")
    print("=" * 60)
    
    # 步骤1: 验证配置
    if not validate_token_ids(inferencer):
        print("❌ Token ID验证失败，请检查配置")
        return False
    
    # 步骤2: 测试token预测
    if not test_token_prediction(inferencer):
        print("❌ Token预测测试失败")
        return False
    
    # 步骤3: 快速生成测试
    if not quick_test_unified_generation(inferencer):
        print("❌ 统一生成测试失败")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！统一多模态生成功能正常工作")
    return True

# 使用示例
if __name__ == "__main__":
    print("统一多模态生成 - 快速测试")
    print("请确保已正确加载模型后调用 comprehensive_test(inferencer)")
    
    # 示例调用代码（需要实际的inferencer实例）:
    """
    # 假设你已经有一个配置好的inferencer
    # inferencer = your_loaded_inferencer
    # comprehensive_test(inferencer)
    """
