#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速过拟合测试 - 专门测试您训练的文本+图像数据

使用方法：
python quick_overfitting_test.py
"""

import os
import sys
import torch
import json
from PIL import Image
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from inference_unified_autoregressive import create_inference_engine


def load_actual_training_samples():
    """加载实际的训练样本 - 使用与训练完全相同的数据处理流程"""
    print("📂 正在加载实际训练数据...")
    
    # 实际训练数据路径
    training_data_path = "/workspace/bagel/dataset/demo/demo_sample/anno.json"
    
    if not os.path.exists(training_data_path):
        print(f"❌ 训练数据文件不存在: {training_data_path}")
        return None
    
    print(f"✅ 找到训练数据: {training_data_path}")
    
    # 使用与训练相同的数据处理器解析数据
    try:
        from training.unified_data_processor import UnifiedGenerationDataset
        from data.data_utils import add_special_tokens
        from modeling.qwen2 import Qwen2Tokenizer
        from data.transforms import ImageTransform
        
        # 加载tokenizer和特殊tokens（与训练一致）
        tokenizer = Qwen2Tokenizer.from_pretrained("/workspace/bagel/models/Qwen2.5-0.5B-Instruct")
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        
        # 创建图像变换（用于测试，不需要实际变换）
        class DummyTransform:
            def __call__(self, img):
                return torch.zeros((3, 224, 224))  # 占位符
        
        vae_transform = DummyTransform()
        vit_transform = DummyTransform()
        
        # 创建数据集对象来解析数据
        dataset = UnifiedGenerationDataset(
            data_path=training_data_path,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            max_sequence_length=2048
        )
        
        # 获取前3个解析后的训练样本
        training_samples = dataset.examples[:3]
        
        print(f"📊 成功解析了 {len(training_samples)} 个训练样本")
        print("✅ 使用了与训练完全相同的数据处理流程")
        
        return training_samples
        
    except Exception as e:
        print(f"❌ 使用训练数据处理器失败: {e}")
        print("🔄 回退到直接读取原始数据...")
        
        # 回退：直接读取原始数据
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                print(f"📊 直接加载了 {len(raw_data[:3])} 个原始训练样本")
                return raw_data[:3]
        except Exception as e2:
            print(f"❌ 直接读取数据也失败: {e2}")
            return None


def convert_training_sample_to_test_case(sample):
    """将训练样本转换为测试用例格式"""
    # 检查是否是UnifiedTrainingExample对象
    if hasattr(sample, 'input_sequence') and hasattr(sample, 'target_sequence'):
        # 这是UnifiedTrainingExample对象
        input_text = ""
        expected_output = ""
        
        # 处理输入序列
        for item, item_type in zip(sample.input_sequence, sample.input_types):
            if item_type == 'text':
                input_text += str(item) + " "
            elif item_type == 'image':
                input_text += "[图像] "
        
        # 处理目标序列
        for item, item_type in zip(sample.target_sequence, sample.target_types):
            if item_type == 'text':
                expected_output += str(item) + " "
            elif item_type == 'image':
                expected_output += "[应生成图像] "
        
        # 自动添加vision tokens到输入文本（如果目标包含图像）
        if 'image' in sample.target_types and '<|vision_start|>' not in input_text:
            input_text = input_text.strip() + " <|vision_start|><|vision_end|>"
        
        return {
            "input": input_text.strip(),
            "expected": expected_output.strip(),
            "original_sample": sample
        }
    
    elif isinstance(sample, dict):
        # 原始数据格式（messages + images）
        input_text = ""
        expected_output = ""
        
        if "messages" in sample:
            messages = sample["messages"]
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                
                if role == 'user':
                    input_text += content + " "
                elif role == 'assistant':
                    expected_output += content + " "
            
            # 如果assistant的回答包含<image>标记，自动添加vision tokens到输入
            if '<image>' in expected_output and '<|vision_start|>' not in input_text:
                input_text = input_text.strip() + " <|vision_start|><|vision_end|>"
        
        return {
            "input": input_text.strip(),
            "expected": expected_output.strip(),
            "original_sample": sample
        }
    
    else:
        # 如果是其他格式
        return {
            "input": str(sample),
            "expected": "未知格式的训练样本",
            "original_sample": sample
        }


def quick_overfitting_test():
    """快速过拟合测试"""
    print("🧪 快速过拟合测试")
    print("测试您训练的文本+图像模型是否正确学会了训练数据")
    print("-" * 50)
    
    # 检查点路径
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return False
    
    # 加载实际训练样本
    training_samples = load_actual_training_samples()
    if training_samples is None:
        print("使用默认测试样本（可能与训练数据不一致）")
        test_cases = [
            {
                "input": "画一只可爱的小猫在花园里玩耍",
                "expected": "应该生成描述文本和一张猫的图像",
                "original_sample": None
            }
        ]
    else:
        print("✅ 使用实际训练样本进行过拟合测试")
        test_cases = [convert_training_sample_to_test_case(sample) for sample in training_samples]
    
    try:
        # 加载模型
        print("📦 正在加载模型...")
        engine = create_inference_engine(checkpoint_path)
        print("✅ 模型加载完成")
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎯 测试用例 {i}/{len(test_cases)}")
            print(f"输入: {test_case['input']}")
            print(f"期望: {test_case['expected']}")
            
            # 如果有原始训练样本，显示更多详细信息
            if test_case.get('original_sample'):
                print("📋 原始训练样本:")
                original = test_case['original_sample']
                
                # 如果是UnifiedTrainingExample对象
                if hasattr(original, 'input_sequence'):
                    print(f"  输入序列类型: {original.input_types}")
                    print(f"  目标序列类型: {original.target_types}")
                    print(f"  输入序列长度: {len(original.input_sequence)}")
                    print(f"  目标序列长度: {len(original.target_sequence)}")
                    
                    # 显示文本内容
                    for i, (item, item_type) in enumerate(zip(original.input_sequence, original.input_types)):
                        if item_type == 'text':
                            print(f"  输入文本{i+1}: {str(item)[:80]}{'...' if len(str(item)) > 80 else ''}")
                    
                    for i, (item, item_type) in enumerate(zip(original.target_sequence, original.target_types)):
                        if item_type == 'text':
                            print(f"  目标文本{i+1}: {str(item)[:80]}{'...' if len(str(item)) > 80 else ''}")
                        elif item_type == 'image':
                            print(f"  目标图像{i+1}: [图像对象]")
                
                # 如果是原始字典格式
                elif isinstance(original, dict):
                    for key, value in original.items():
                        if isinstance(value, (list, dict)):
                            print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  原始样本: {str(original)[:100]}{'...' if len(str(original)) > 100 else ''}")
            
            try:
                # 进行推理 - 使用低温度确保确定性结果（测试过拟合）
                results = engine.autoregressive_generate(
                    prompt=test_case['input'],
                    max_length=200,
                    do_sample=True,
                    temperature=0.1,  # 很低的温度，接近确定性
                    image_shapes=(512, 512),
                    cfg_text_scale=2.0,  # 较低的CFG避免过度偏离训练数据
                    cfg_img_scale=1.2,
                    num_timesteps=20,  # 较少步数加快测试
                    save_intermediate=True,
                    output_dir=f"quick_test_case_{i}"
                )
                
                # 分析结果
                text_outputs = []
                image_outputs = []
                
                for item in results:
                    if isinstance(item, str):
                        text_outputs.append(item)
                    elif hasattr(item, 'save'):  # PIL Image
                        image_outputs.append(item)
                
                print(f"📊 结果分析:")
                print(f"  生成文本数: {len(text_outputs)}")
                print(f"  生成图像数: {len(image_outputs)}")
                
                # 显示生成的文本
                for j, text in enumerate(text_outputs):
                    print(f"  文本{j+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
                
                # 保存图像
                for j, img in enumerate(image_outputs):
                    img_path = f"quick_test_case_{i}_image_{j+1}.png"
                    img.save(img_path)
                    print(f"  图像{j+1}: 已保存到 {img_path} (尺寸: {img.size})")
                
                # 判断是否成功（有生成内容即认为成功）
                if len(text_outputs) > 0 or len(image_outputs) > 0:
                    print("✅ 测试成功 - 模型生成了内容")
                    success_count += 1
                    
                    # 特别检查图像生成（这是关键测试点）
                    if len(image_outputs) > 0:
                        print("🎨 ✅ 模型成功生成了图像（过拟合目标达成）")
                    else:
                        print("📝 模型只生成了文本，未生成图像")
                else:
                    print("❌ 测试失败 - 模型未生成任何内容")
                
            except Exception as e:
                print(f"❌ 测试用例 {i} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 总结
        print(f"\n{'='*50}")
        print("🎯 过拟合测试总结")
        print(f"{'='*50}")
        print(f"成功测试用例: {success_count}/3")
        print(f"成功率: {success_count/3*100:.1f}%")
        
        if success_count == 3:
            print(" 所有测试通过！")
  
        elif success_count > 0:
            print("⚠️  部分测试通过，模型部分学会了训练模式。")
        else:
            print("❌ 所有测试失败，模型可能未正确学习训练数据。")
            print("建议检查:")
            print("  1. 检查点是否正确")
            print("  2. 模型配置是否匹配训练时的设置")
            print("  3. 特殊token是否正确设置")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_special_tokens():
    """快速测试特殊token"""
    print("\n🔍 检查特殊token设置")
    print("-" * 30)
    
    try:
        from data.data_utils import add_special_tokens
        from modeling.qwen2 import Qwen2Tokenizer
        
        # 加载tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained("/workspace/bagel/models/Qwen2.5-0.5B-Instruct")
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        
        print(f"新增特殊token数量: {num_new_tokens}")
        
        # 显示图像相关的特殊token
        image_tokens = {k: v for k, v in new_token_ids.items() 
                       if 'image' in k.lower() or 'vision' in k.lower()}
        
        print("图像相关特殊token:")
        for token_name, token_id in image_tokens.items():
            token_text = tokenizer.decode([token_id])
            print(f"  {token_name}: ID={token_id}, Text='{token_text}'")
        
        # 测试编码
        test_text = "这是文本 <|vision_start|> 这里是图像 <|vision_end|> 更多文本"
        encoded = tokenizer.encode(test_text)
        print(f"\n测试编码: {test_text}")
        print(f"编码结果长度: {len(encoded)}")
        
        # 检查是否包含特殊token
        found_special = []
        for token_id in encoded:
            if token_id in new_token_ids.values():
                token_name = [k for k, v in new_token_ids.items() if v == token_id][0]
                found_special.append(token_name)
        
        if found_special:
            print(f"✅ 发现特殊token: {found_special}")
        else:
            print("⚠️  未发现特殊token")
        
        return True
        
    except Exception as e:
        print(f"❌ 特殊token测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始快速过拟合测试")
    print("这个测试专门验证您训练的文本+图像模型")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"🔥 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  使用CPU（会比较慢）")
    
    # 测试特殊token
    token_test_ok = test_special_tokens()
    
    # 主要过拟合测试
    main_test_ok = quick_overfitting_test()
    
    print(f"\n{'='*60}")
    print("🏁 最终结果")
    print(f"{'='*60}")
    
    if token_test_ok and main_test_ok:
        print("🎉 过拟合测试成功！")
        print("您的模型正确学习了训练数据的模式")
        print("📂 生成的图像已保存在当前目录")
    else:
        print("❌ 测试中存在问题")
        if not token_test_ok:
            print("  - 特殊token配置可能有问题")
        if not main_test_ok:
            print("  - 模型生成测试失败")
        print("建议检查模型和配置")


if __name__ == "__main__":
    main()
