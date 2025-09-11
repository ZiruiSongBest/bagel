#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速推理脚本 - 用于快速测试您训练的自回归交错生成模型

使用方法：
python quick_inference.py --prompt "画一只猫在花园里玩耍"
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from inference_unified_autoregressive import create_inference_engine


def quick_test(checkpoint_path, prompt, output_dir="quick_test_output"):
    """快速测试函数"""
    print(f"🚀 快速推理测试")
    print(f"检查点: {checkpoint_path}")
    print(f"提示: {prompt}")
    print("-" * 50)
    
    try:
        # 检查检查点是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点路径不存在: {checkpoint_path}")
            print("请检查路径是否正确")
            return False
        
        # 创建推理引擎
        print("📦 正在加载模型...")
        engine = create_inference_engine(checkpoint_path)
        print("✅ 模型加载完成")
        
        # 执行推理
        print("🎯 开始生成...")
        results = engine.autoregressive_generate(
            prompt=prompt,
            max_length=200,
            do_sample=True,
            temperature=0.8,
            image_shapes=(512, 512),  # 使用较小尺寸加快速度
            save_intermediate=True,
            output_dir=output_dir
        )
        
        # 显示结果
        print(f"✅ 生成完成！共生成 {len(results)} 个项目")
        
        text_count = 0
        image_count = 0
        
        for i, item in enumerate(results):
            if isinstance(item, str):
                text_count += 1
                print(f"📝 [{i}] 文本: {item[:100]}{'...' if len(item) > 100 else ''}")
            elif isinstance(item, Image.Image):
                image_count += 1
                print(f"🖼️  [{i}] 图像: {item.size}")
        
        print(f"\n📊 统计: {text_count} 个文本, {image_count} 个图像")
        print(f"💾 结果已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="快速推理测试")
    
    parser.add_argument("--checkpoint_path", type=str,
                       default="/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800",
                       help="检查点路径")
    parser.add_argument("--prompt", type=str,
                       default="画一只可爱的小猫在花园里玩耍，阳光明媚",
                       help="输入提示")
    parser.add_argument("--output_dir", type=str, default="quick_test_output",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"🔥 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  使用CPU（可能较慢）")
    
    # 执行快速测试
    success = quick_test(args.checkpoint_path, args.prompt, args.output_dir)
    
    if success:
        print("\n🎉 快速测试成功！")
        print("您可以查看输出目录中的生成结果")
    else:
        print("\n💥 快速测试失败")
        print("请检查模型路径和环境配置")


if __name__ == "__main__":
    main()
