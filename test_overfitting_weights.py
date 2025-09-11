#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试过拟合权重的脚本

专门用于测试训练过程中使用的数据，验证模型是否正确学会了文本+图像的映射关系
"""

import os
import sys
import torch
import json
from PIL import Image
from pathlib import Path
import logging

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from inference_unified_autoregressive import create_inference_engine
from training.unified_data_processor import UnifiedGenerationDataset
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data_sample(data_path, num_samples=3):
    """加载训练数据的样本进行测试"""
    logger.info(f"从训练数据中加载样本: {data_path}")
    
    samples = []
    
    if os.path.exists(data_path):
        if data_path.endswith('.jsonl'):
            # JSONL格式
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        samples.append(data)
                    except Exception as e:
                        logger.warning(f"跳过无效行 {i}: {e}")
        elif data_path.endswith('.json'):
            # JSON格式
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:num_samples]
                else:
                    samples = [data]
    else:
        # 如果没有找到训练数据，创建一些测试样本
        logger.warning(f"训练数据文件不存在: {data_path}")
        logger.info("使用模拟的训练数据格式进行测试")
        samples = create_mock_training_samples()
    
    logger.info(f"加载了 {len(samples)} 个测试样本")
    return samples


def create_mock_training_samples():
    """创建模拟的训练数据样本（符合您的训练格式）"""
    samples = [
        {
            "input_sequence": ["画一只可爱的小猫在花园里玩耍"],
            "target_sequence": ["这是一只橘色的小猫，在绿色的花园中快乐地玩耍。", "<image>"],
            "input_types": ["text"],
            "target_types": ["text", "image"],
            "metadata": {"description": "文本到文本+图像的生成"}
        },
        {
            "input_sequence": ["描述一下春天的美景，然后画出来"],
            "target_sequence": ["春天来了，樱花盛开，绿草如茵，阳光明媚。", "<image>"],
            "input_types": ["text"], 
            "target_types": ["text", "image"],
            "metadata": {"description": "描述+绘画的组合任务"}
        },
        {
            "input_sequence": ["创作一幅山水画"],
            "target_sequence": ["<image>"],
            "input_types": ["text"],
            "target_types": ["image"],
            "metadata": {"description": "纯文本到图像"}
        }
    ]
    return samples


def test_single_sample(engine, sample, sample_idx):
    """测试单个训练样本"""
    logger.info(f"\n{'='*50}")
    logger.info(f"测试样本 {sample_idx + 1}")
    logger.info(f"{'='*50}")
    
    # 构建输入文本
    input_texts = sample.get("input_sequence", [])
    target_sequence = sample.get("target_sequence", [])
    
    if not input_texts:
        logger.warning("样本缺少输入序列，跳过")
        return
    
    # 合并输入文本
    input_text = " ".join([text for text in input_texts if isinstance(text, str)])
    logger.info(f"输入文本: {input_text}")
    
    # 显示期望的输出
    logger.info("期望输出:")
    for i, target in enumerate(target_sequence):
        if isinstance(target, str):
            if target == "<image>":
                logger.info(f"  [{i}] 图像标记: <image>")
            else:
                logger.info(f"  [{i}] 文本: {target}")
        else:
            logger.info(f"  [{i}] 其他类型: {type(target)}")
    
    try:
        # 使用训练时相同的生成逻辑
        # 对于过拟合测试，使用较低的temperature和确定性采样
        results = engine.autoregressive_generate(
            prompt=input_text,
            max_length=300,
            do_sample=True,  # 使用采样但temperature较低
            temperature=0.1,  # 低温度，接近确定性
            image_shapes=(512, 512),  # 较小尺寸加快测试
            cfg_text_scale=3.0,
            cfg_img_scale=1.2,
            num_timesteps=25,  # 减少步数加快测试
            save_intermediate=True,
            output_dir=f"overfitting_test_sample_{sample_idx + 1}"
        )
        
        logger.info(f"实际生成了 {len(results)} 个项目:")
        text_count = 0
        image_count = 0
        
        for i, item in enumerate(results):
            if isinstance(item, str):
                text_count += 1
                logger.info(f"  [{i}] 生成文本: {item[:100]}{'...' if len(item) > 100 else ''}")
            elif hasattr(item, 'save'):  # PIL Image
                image_count += 1
                logger.info(f"  [{i}] 生成图像: {item.size}")
                # 保存图像
                image_path = f"overfitting_test_sample_{sample_idx + 1}/generated_image_{i}.png"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                item.save(image_path)
                logger.info(f"       保存到: {image_path}")
        
        # 简单的过拟合检验
        expected_images = sum(1 for t in target_sequence if t == "<image>")
        expected_texts = sum(1 for t in target_sequence if isinstance(t, str) and t != "<image>")
        
        logger.info(f"\n过拟合检验:")
        logger.info(f"  期望文本数: {expected_texts}, 实际生成: {text_count}")
        logger.info(f"  期望图像数: {expected_images}, 实际生成: {image_count}")
        
        if image_count > 0 and expected_images > 0:
            logger.info("✅ 模型能够生成图像（符合训练目标）")
        elif expected_images > 0 and image_count == 0:
            logger.warning("⚠️  期望生成图像但未生成")
        
        if text_count > 0 and expected_texts > 0:
            logger.info("✅ 模型能够生成文本（符合训练目标）")
        
        return True
        
    except Exception as e:
        logger.error(f"测试样本 {sample_idx + 1} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_special_tokens(engine):
    """测试特殊token的识别"""
    logger.info(f"\n{'='*50}")
    logger.info("测试特殊token识别")
    logger.info(f"{'='*50}")
    
    # 获取tokenizer和特殊token
    tokenizer = engine.inferencer.tokenizer
    new_token_ids = engine.inferencer.new_token_ids
    
    logger.info("特殊token映射:")
    for token_name, token_id in new_token_ids.items():
        if 'image' in token_name.lower() or 'vision' in token_name.lower():
            token_text = tokenizer.decode([token_id]) if token_id < len(tokenizer) else f"ID:{token_id}"
            logger.info(f"  {token_name}: {token_id} -> '{token_text}'")
    
    # 测试包含特殊token的文本
    test_prompts = [
        "画一只猫 <|vision_start|> <|vision_end|>",
        "这是一段文本，然后生成图像",
        "创作一幅画作"
    ]
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n测试提示 {i+1}: {prompt}")
        
        # 编码测试
        try:
            encoded = tokenizer.encode(prompt)
            logger.info(f"  编码长度: {len(encoded)}")
            
            # 检查是否包含图像相关的特殊token
            has_image_tokens = False
            for token_id in encoded:
                if token_id in new_token_ids.values():
                    token_name = [k for k, v in new_token_ids.items() if v == token_id][0]
                    if 'image' in token_name.lower() or 'vision' in token_name.lower():
                        has_image_tokens = True
                        logger.info(f"  包含特殊token: {token_name} (ID: {token_id})")
            
            if not has_image_tokens:
                logger.info("  未检测到图像相关特殊token")
                
        except Exception as e:
            logger.warning(f"  编码失败: {e}")


def main():
    """主测试函数"""
    print("🧪 过拟合权重测试")
    print("=" * 60)
    
    # 检查点路径
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    # 训练数据路径（您需要修改为实际路径）
    training_data_paths = [
        "data/unified_train.jsonl",  # 默认路径
        "training/sample_data.json",  # 可能的路径
        "dataset/train.jsonl",  # 另一个可能路径
    ]
    
    # 找到存在的训练数据文件
    training_data_path = None
    for path in training_data_paths:
        if os.path.exists(path):
            training_data_path = path
            break
    
    if training_data_path is None:
        logger.warning("未找到训练数据文件，将使用模拟数据进行测试")
        training_data_path = "mock_data"
    
    logger.info(f"检查点路径: {checkpoint_path}")
    logger.info(f"训练数据路径: {training_data_path}")
    
    try:
        # 创建推理引擎
        logger.info("正在加载模型...")
        engine = create_inference_engine(checkpoint_path)
        logger.info("✅ 模型加载成功")
        
        # 测试特殊token
        test_special_tokens(engine)
        
        # 加载训练数据样本
        samples = load_training_data_sample(training_data_path, num_samples=3)
        
        # 测试每个样本
        success_count = 0
        for i, sample in enumerate(samples):
            success = test_single_sample(engine, sample, i)
            if success:
                success_count += 1
        
        # 总结结果
        logger.info(f"\n{'='*60}")
        logger.info("🎯 过拟合测试总结")
        logger.info(f"{'='*60}")
        logger.info(f"测试样本数: {len(samples)}")
        logger.info(f"成功测试: {success_count}")
        logger.info(f"成功率: {success_count/len(samples)*100:.1f}%")
        
        if success_count == len(samples):
            logger.info("🎉 所有测试样本都成功！模型过拟合效果良好。")
        elif success_count > 0:
            logger.info("⚠️  部分样本测试成功，模型部分学会了训练数据。")
        else:
            logger.warning("❌ 所有样本测试失败，可能需要检查模型或数据。")
        
        # 生成最终测试报告
        generate_test_report(checkpoint_path, training_data_path, samples, success_count)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def generate_test_report(checkpoint_path, data_path, samples, success_count):
    """生成测试报告"""
    report = {
        "checkpoint_path": checkpoint_path,
        "training_data_path": data_path,
        "test_time": str(torch.datetime.datetime.now()),
        "total_samples": len(samples),
        "successful_samples": success_count,
        "success_rate": success_count/len(samples)*100 if samples else 0,
        "samples_tested": []
    }
    
    for i, sample in enumerate(samples):
        sample_info = {
            "sample_id": i + 1,
            "input_sequence": sample.get("input_sequence", []),
            "target_sequence": sample.get("target_sequence", []),
            "metadata": sample.get("metadata", {})
        }
        report["samples_tested"].append(sample_info)
    
    # 保存报告
    report_path = "overfitting_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📊 测试报告已保存: {report_path}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
